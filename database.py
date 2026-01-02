"""
DynamoDB integration module for photo storage.
Handles storing and retrieving photos as base64, along with metadata.
"""

import base64
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
import os

# Try to import boto3, but allow graceful fallback
try:
    import boto3
    from botocore.exceptions import ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    ClientError = Exception  # Placeholder


def is_dynamodb_available():
    """Check if DynamoDB support is available (boto3 installed)."""
    return BOTO3_AVAILABLE


class PhotoDatabase:
    """Handles all DynamoDB operations for photo storage."""
    
    def __init__(
        self,
        table_name: str = "PhotoStorageApp",
        region_name: str = "us-east-1",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        use_local: bool = False
    ):
        """
        Initialize DynamoDB connection.
        
        Args:
            table_name: Name of the DynamoDB table
            region_name: AWS region
            aws_access_key_id: AWS access key (optional, uses env vars if not provided)
            aws_secret_access_key: AWS secret key (optional, uses env vars if not provided)
            use_local: If True, use local DynamoDB instance
        """
        if not BOTO3_AVAILABLE:
            raise ImportError(
                "boto3 is required for DynamoDB support. "
                "Install it with: pip install boto3"
            )
        
        self.table_name = table_name
        
        if use_local:
            self.dynamodb = boto3.resource(
                'dynamodb',
                endpoint_url='http://localhost:8000',
                region_name=region_name,
                aws_access_key_id='dummy',
                aws_secret_access_key='dummy'
            )
        else:
            session_kwargs = {'region_name': region_name}
            if aws_access_key_id and aws_secret_access_key:
                session_kwargs['aws_access_key_id'] = aws_access_key_id
                session_kwargs['aws_secret_access_key'] = aws_secret_access_key
            
            self.dynamodb = boto3.resource('dynamodb', **session_kwargs)
        
        self.table = self.dynamodb.Table(table_name)
    
    def create_table_if_not_exists(self) -> bool:
        """
        Create the photos table if it doesn't exist.
        
        Returns:
            True if table was created, False if it already existed
        """
        try:
            # Check if table exists
            self.table.load()
            return False
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceNotFoundException':
                # Create table
                table = self.dynamodb.create_table(
                    TableName=self.table_name,
                    KeySchema=[
                        {'AttributeName': 'photo_id', 'KeyType': 'HASH'}
                    ],
                    AttributeDefinitions=[
                        {'AttributeName': 'photo_id', 'AttributeType': 'S'}
                    ],
                    BillingMode='PAY_PER_REQUEST'
                )
                table.wait_until_exists()
                self.table = table
                return True
            raise
    
    def save_photo(
        self,
        photo_id: str,
        image_base64: str,
        filename: str,
        orientation: int = 0,
        tags: Optional[List[str]] = None,
        is_favorite: bool = False,
        faces: Optional[List[Dict[str, Any]]] = None,
        thumbnail_base64: Optional[str] = None,
        original_base64: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Save a photo to DynamoDB.
        
        Args:
            photo_id: Unique identifier for the photo
            image_base64: Base64 encoded image data (cropped/processed version)
            filename: Original filename
            orientation: Rotation in degrees (0, 90, 180, 270)
            tags: List of tags for searching
            is_favorite: Whether photo is starred
            faces: List of face data with encodings and names
            thumbnail_base64: Base64 encoded thumbnail
            original_base64: Base64 encoded original image before cropping
            
        Returns:
            The saved item
        """
        timestamp = datetime.utcnow().isoformat()
        
        item = {
            'photo_id': photo_id,
            'image_base64': image_base64,
            'filename': filename,
            'orientation': orientation,
            'tags': tags or [],
            'is_favorite': is_favorite,
            'faces': json.dumps(faces) if faces else '[]',
            'created_at': timestamp,
            'updated_at': timestamp
        }
        
        if thumbnail_base64:
            item['thumbnail_base64'] = thumbnail_base64
        
        if original_base64:
            item['original_base64'] = original_base64
        
        self.table.put_item(Item=item)
        return item
    
    def get_photo(self, photo_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a single photo by ID.
        
        Args:
            photo_id: The photo's unique identifier
            
        Returns:
            Photo data or None if not found
        """
        try:
            response = self.table.get_item(Key={'photo_id': photo_id})
            item = response.get('Item')
            if item and 'faces' in item:
                item['faces'] = json.loads(item['faces'])
            return item
        except ClientError:
            return None
    
    def get_all_photos(self) -> List[Dict[str, Any]]:
        """
        Retrieve all photos from the database.
        
        Returns:
            List of all photo items
        """
        items = []
        try:
            response = self.table.scan()
            items.extend(response.get('Items', []))
            
            # Handle pagination
            while 'LastEvaluatedKey' in response:
                response = self.table.scan(
                    ExclusiveStartKey=response['LastEvaluatedKey']
                )
                items.extend(response.get('Items', []))
            
            # Parse faces JSON
            for item in items:
                if 'faces' in item:
                    item['faces'] = json.loads(item['faces'])
            
            # Sort by created_at descending
            items.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            
            return items
        except ClientError:
            return []
    
    def get_favorites(self) -> List[Dict[str, Any]]:
        """
        Retrieve all favorited photos.
        
        Returns:
            List of favorited photo items
        """
        all_photos = self.get_all_photos()
        return [p for p in all_photos if p.get('is_favorite', False)]
    
    def search_by_tags(self, search_tags: List[str]) -> List[Dict[str, Any]]:
        """
        Search photos by tags.
        
        Args:
            search_tags: List of tags to search for
            
        Returns:
            List of matching photos
        """
        all_photos = self.get_all_photos()
        search_tags_lower = [t.lower() for t in search_tags]
        
        matching = []
        for photo in all_photos:
            photo_tags = [t.lower() for t in photo.get('tags', [])]
            if any(st in photo_tags for st in search_tags_lower):
                matching.append(photo)
        
        return matching
    
    def search_by_person(self, person_name: str) -> List[Dict[str, Any]]:
        """
        Search photos containing a specific person.
        
        Args:
            person_name: Name of the person to search for
            
        Returns:
            List of photos containing that person
        """
        all_photos = self.get_all_photos()
        person_name_lower = person_name.lower()
        
        matching = []
        for photo in all_photos:
            faces = photo.get('faces', [])
            for face in faces:
                if face.get('name', '').lower() == person_name_lower:
                    matching.append(photo)
                    break
        
        return matching
    
    def update_photo(
        self,
        photo_id: str,
        orientation: Optional[int] = None,
        tags: Optional[List[str]] = None,
        is_favorite: Optional[bool] = None,
        faces: Optional[List[Dict[str, Any]]] = None,
        image_base64: Optional[str] = None
    ) -> bool:
        """
        Update photo metadata.
        
        Args:
            photo_id: Photo to update
            orientation: New orientation (if provided)
            tags: New tags (if provided)
            is_favorite: New favorite status (if provided)
            faces: New face data (if provided)
            image_base64: New image data (if provided)
            
        Returns:
            True if successful
        """
        update_expr_parts = ['#updated_at = :updated_at']
        expr_attr_names = {'#updated_at': 'updated_at'}
        expr_attr_values = {':updated_at': datetime.utcnow().isoformat()}
        
        if orientation is not None:
            update_expr_parts.append('#orientation = :orientation')
            expr_attr_names['#orientation'] = 'orientation'
            expr_attr_values[':orientation'] = orientation
        
        if tags is not None:
            update_expr_parts.append('#tags = :tags')
            expr_attr_names['#tags'] = 'tags'
            expr_attr_values[':tags'] = tags
        
        if is_favorite is not None:
            update_expr_parts.append('#is_favorite = :is_favorite')
            expr_attr_names['#is_favorite'] = 'is_favorite'
            expr_attr_values[':is_favorite'] = is_favorite
        
        if faces is not None:
            update_expr_parts.append('#faces = :faces')
            expr_attr_names['#faces'] = 'faces'
            expr_attr_values[':faces'] = json.dumps(faces)
        
        if image_base64 is not None:
            update_expr_parts.append('#image_base64 = :image_base64')
            expr_attr_names['#image_base64'] = 'image_base64'
            expr_attr_values[':image_base64'] = image_base64
        
        try:
            self.table.update_item(
                Key={'photo_id': photo_id},
                UpdateExpression='SET ' + ', '.join(update_expr_parts),
                ExpressionAttributeNames=expr_attr_names,
                ExpressionAttributeValues=expr_attr_values
            )
            return True
        except ClientError:
            return False
    
    def delete_photo(self, photo_id: str) -> bool:
        """
        Delete a photo.
        
        Args:
            photo_id: Photo to delete
            
        Returns:
            True if successful
        """
        try:
            self.table.delete_item(Key={'photo_id': photo_id})
            return True
        except ClientError:
            return False
    
    def get_all_known_faces(self) -> Dict[str, List[List[float]]]:
        """
        Get all named faces across all photos for recognition.
        
        Returns:
            Dictionary mapping names to lists of face encodings
        """
        all_photos = self.get_all_photos()
        known_faces: Dict[str, List[List[float]]] = {}
        
        for photo in all_photos:
            faces = photo.get('faces', [])
            for face in faces:
                name = face.get('name')
                encoding = face.get('encoding')
                if name and encoding:
                    if name not in known_faces:
                        known_faces[name] = []
                    known_faces[name].append(encoding)
        
        return known_faces
    
    def get_all_tags(self) -> List[str]:
        """
        Get all unique tags across all photos.
        
        Returns:
            Sorted list of unique tags
        """
        all_photos = self.get_all_photos()
        all_tags = set()
        
        for photo in all_photos:
            all_tags.update(photo.get('tags', []))
        
        return sorted(list(all_tags))
    
    def get_all_people(self) -> List[str]:
        """
        Get all unique person names across all photos.
        
        Returns:
            Sorted list of unique names
        """
        all_photos = self.get_all_photos()
        all_names = set()
        
        for photo in all_photos:
            faces = photo.get('faces', [])
            for face in faces:
                name = face.get('name')
                if name:
                    all_names.add(name)
        
        return sorted(list(all_names))
