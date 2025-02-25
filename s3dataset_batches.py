
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
from concurrent.futures import ThreadPoolExecutor, as_completed
import botocore.config
import os
import redis
import torch
from io import BytesIO

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


class TensorSockerDataset(Dataset):
    def __init__(self, s3_data_dir: str, 
                 transform=None,
                 cache_address=None,  
                 simulate_mode=False, 
                 simulate_time_for_cache_miss=0,
                 simulate_time_for_cache_hit=0,
                 cache_transformations=True,
                 use_compression=False,
                 use_local_folder=False,
                 ssl=True):
        
        self.s3_bucket = S3Url(s3_data_dir).bucket
        self.s3_prefix = S3Url(s3_data_dir).key
        self.s3_data_dir = s3_data_dir
        self.s3_client = None
        self.transform = transform
        self.using_local_folder = use_local_folder
        self.samples = self._get_sample_list_from_s3()
        self.simulate_mode = simulate_mode
        self._simlute_time_for_cache_miss = simulate_time_for_cache_miss
        self._simlute_time_for_cache_hit = simulate_time_for_cache_hit
        self.cache_transformations = cache_transformations
        self.use_compression = use_compression
        self.ssl = ssl
        self.cache_client = None

        if cache_address is not None:
            self.cache_host, self.cache_port = cache_address.split(":")
            self.cache_port = int(self.cache_port)
            self.use_cache = True
        else:
            self.use_cache = False
    
    
    def __getstate__(self):

        state = self.__dict__.copy()
        del state['cache_client']  # Remove the Redis connection before pickling
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.ssl:
            self.cache_client = redis.StrictRedis(host=self.cache_host, port=self.cache_port, ssl=True)
        else:
            self.cache_client = redis.StrictRedis(host=self.cache_host, port=self.cache_port)

    
    def check_s3_client(self):
        if self.s3_client is None:
            self.s3_client = boto3.client('s3', config=botocore.config.Config(
                max_pool_connections=100))
            

    @functools.cached_property
    def _classed_items(self) -> List[Tuple[str, int]]:
        return [(blob, class_index)
            for class_index, blob_class in enumerate(self.samples)
            for blob in self.samples[blob_class]]
    
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
    
    def __len__(self) -> int:
        return sum(len(class_items) for class_items in self.samples.values())
    

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
        batch_id, batch_indices = idx
        # batch_indices = self._classed_items[idx][0]  # Simplified for demonstration
        start_loading_time = time.perf_counter()
        samples, labels, cache_hit_count = self.fetch_batch_data(batch_indices)
        start_transformation_time  = time.perf_counter()

        start_transformation_time = time.perf_counter()
        if self.transform is not None:
            for i in range(len(samples)):
                samples[i] = self.transform(samples[i])        
        transformation_time =  time.perf_counter() - start_transformation_time
        # Convert to tensors
        samples= torch.stack(samples)
        labels = torch.tensor(labels)
        data_fetch_time  = time.perf_counter() - start_loading_time - transformation_time
        return (samples, labels)

        # return samples, labels,batch_id,data_fetch_time,transformation_time

    def fetch_batch_data(self, batch_indices: List[str]):
        data_samples, labels = [], []
        cache_hits = 0
        if self.using_local_folder:
            data = None
            for idx in batch_indices:
                data_path, label = self._classed_items[idx]
                if self.use_cache:
                    data = self.fetch_item_from_cache(idx)
                if data is not None:
                    cache_hits += 1
                else:
                    data_path = os.path.join('data/cifar10/', data_path) 
                    # with open(data_path, 'rb') as f:
                    data = Image.open(data_path)
                    if self.use_cache:
                        self.put_item_in_cache(idx, data)
                data_samples.append(data.convert("RGB") )
                labels.append(label)
            return data_samples, labels, cache_hits
        else:
            self.check_s3_client()
            with ThreadPoolExecutor() as executor:
                futures = {executor.submit(self.get_data_sample, idx): idx for idx in batch_indices}
                for future in as_completed(futures):
                    data_sample, label, cahe_hit = future.result()
                    if cahe_hit:
                        cache_hits += 1
                    data_samples.append(data_sample)
                    labels.append(label)
            return data_samples, labels, cache_hits
        

    def _initialize_cache_client(self):
        """Initialize Redis cache client if not already connected."""
        if self.cache_client is None:
            # self.cache_client = redis.StrictRedis(host=self.cache_host, port=self.cache_port)
            if self.ssl:
                self.cache_client = redis.StrictRedis(host=self.cache_host, port=self.cache_port, ssl=True)
            else:
                self.cache_client = redis.StrictRedis(host=self.cache_host, port=self.cache_port)

    def fetch_item_from_cache(self, idx: int):
          self._initialize_cache_client()
          byte_image = self.cache_client.get(idx)
          if byte_image is None:
                return None
          byteImgIO = io.BytesIO(byte_image)
          data = Image.open(byteImgIO)
          return data
    
    def put_item_in_cache(self, idx: int, data):
        self._initialize_cache_client()
        # Serialize the image using BytesIO
        img_byte_arr = BytesIO()
        data.save(img_byte_arr, format='PNG')  # Save as PNG to the byte array
        img_byte_arr.seek(0)  # Reset pointer to the start of the byte array
        self.cache_client.set(idx, img_byte_arr.read())

    def get_data_sample(self,idx) -> tuple:  
        data_path, label = self._classed_items[idx]
        cache_hit = False
        if self.use_cache:
            data = self.fetch_item_from_cache(idx)
            cache_hit = True
        else:
            obj = self.s3_client.get_object(Bucket=self.s3_bucket, Key=data_path)
            data = Image.open(BytesIO(obj['Body'].read()))
            if self.use_cache:
                self.put_item_in_cache(idx, data)
        return data.convert("RGB"), label, cache_hit

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
       