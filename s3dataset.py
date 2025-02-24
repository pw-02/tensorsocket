
# No 'default_generator' in torch/__init__.pyi
from typing import TypeVar, List, Tuple, Dict
from datetime import datetime
import random
import PIL.Image as Image
import numpy as np
import io
import numpy as np
import PIL
from urllib.parse import urlparse
import boto3
import functools
from torch.utils.data import Dataset
import json
import time

class S3Url(object):
    def __init__(self, url):
        self._parsed = urlparse(url, allow_fragments=False)

    @property
    def bucket(self):
        return self._parsed.netloc

    @property
    def key(self):
        if self._parsed.query:
            return self._parsed.path.lstrip('/') + '?' + self._parsed.query
        else:
            return self._parsed.path.lstrip('/')

    @property
    def url(self):
        return self._parsed.geturl()


class S3ImageDataset(Dataset):
    def __init__(self, s3_data_dir: str, transform=None):
        self.s3_bucket = S3Url(s3_data_dir).bucket
        self.s3_prefix = S3Url(s3_data_dir).key
        self.s3_data_dir = s3_data_dir
        self.transform = transform
        self.samples = self._get_sample_list_from_s3(use_index_file=True, images_only=True)
        self.s3_client = None
        # self.s3_client = boto3.client('s3')
    
    @functools.cached_property
    def _classed_items(self) -> List[Tuple[str, int]]:
        return [(blob, class_index)
            for class_index, blob_class in enumerate(self.samples)
            for blob in self.samples[blob_class]]


    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target, index) where target is class_index of the target class.
        """
        if self.s3_client is None:
            self.s3_client = boto3.client('s3')
        path, label = self._classed_items[index]
        insertion_time = datetime.now()
        insertion_time = insertion_time.strftime("%H:%M:%S")
        # print("train_search_index: %d time: %s" % (index, insertion_time))
        sample = self.fetch_image_from_s3(path)
        sample = sample.convert('RGB')
        
        transform_start_time = time.perf_counter()
        if self.transform is not None:
            sample = self.transform(sample)
        transform_duration = time.perf_counter() - transform_start_time
        return sample, label

    def __len__(self) -> int:
        return sum(len(class_items) for class_items in self.samples.values())
    
    def fetch_image_from_s3(self, data_path):
        if self.s3_client is None:
            self.s3_client = boto3.client('s3')
        obj = self.s3_client.get_object(Bucket=self.s3_bucket, Key=data_path)
        img_data = obj['Body'].read()
        image = Image.open(io.BytesIO(img_data)) #.convert('RGB')
        return image
    
    def _get_sample_list_from_s3(self, use_index_file=True, images_only=True) -> Dict[str, List[str]]:
        s3_client = boto3.client('s3')

        index_file_key = f"{self.s3_prefix}_paired_index.json"
        paired_samples = {}

        if use_index_file:
            try:
                index_object = s3_client.get_object(Bucket=self.s3_bucket, Key=index_file_key)
                file_content = index_object['Body'].read().decode('utf-8')
                paired_samples = json.loads(file_content)
                return paired_samples
            except Exception as e:
                print(f"Error reading index file '{index_file_key}': {e}")

        paginator = s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=self.s3_bucket, Prefix=self.s3_prefix):
            for blob in page.get('Contents', []):
                blob_path = blob.get('Key')
                
                if blob_path.endswith("/"):
                    continue  # Skip folders
                
                stripped_path = blob_path[len(self.s3_prefix):].lstrip("/")
                if stripped_path == blob_path:
                    continue  # No matching prefix, skip

                if images_only and not blob_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue  # Skip non-image files
                
                if 'index.json' in blob_path:
                    continue  # Skip index file

                blob_class = stripped_path.split("/")[0]
                if blob_class not in paired_samples:
                    paired_samples[blob_class] = []
                paired_samples[blob_class].append(blob_path)

        if use_index_file and paired_samples:
            s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=index_file_key,
                Body=json.dumps(paired_samples, indent=4).encode('utf-8')
            )

        return paired_samples
       