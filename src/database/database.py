# general
import os
import sys
import re
import json
import pandas as pd

# database
from chromadb import Client, PersistentClient

class LocalDatabase:
    def __init__(self, db_dir: str):
        # create database directory if it doesn't exist
        abs_db_dir = os.path.abspath(db_dir)
        if not os.path.exists(abs_db_dir) or not os.path.isdir(abs_db_dir):
            os.makedirs(abs_db_dir)

        # create variables
        self.db_path = abs_db_dir
        self.db_name = "local_database"
        # create persistent client
        self.client = PersistentClient(self.db_path)

    def create_collection_from_image_metadata(self, source_dir_list: list, item_state_baseline_path: str, use_labeled: bool=True):
        # input handling
        # item-state baseline path
        abs_item_state_baseline_path = os.path.abspath(item_state_baseline_path)
        if not os.path.exists(abs_item_state_baseline_path) or not os.path.splitext(abs_item_state_baseline_path)[1] == '.json':
            print("Item-state baseline json file was not found.")
            return
        # source directory list
        if not source_dir_list:
            print("No source directories were passed.")
            return
        # check source paths
        self.sources = []
        for source_path in source_dir_list:
            abs_source_path = os.path.abspath(source_path)
            if not os.path.exists(abs_source_path) or not os.path.isdir(abs_source_path):
                print(f"Source path {abs_source_path} does not exist.")
                continue
            self.sources.append(abs_source_path)
        if not self.sources:
            print("None of the source paths exist.")
            return
        
        # create collection
        self.collection = self.client.get_or_create_collection(name=self.db_name)
        # add item-state baseline to collection
        with open(abs_item_state_baseline_path, 'r') as f:
            self.item_state_baseline = json.load(f)
        self.collection.add(
            documents=json.dumps(self.item_state_baseline),
            ids='item_state_baseline'
        )

        # iterate through source directories
        for source_path in self.sources:
            self.add_folder_to_collection(
                source_dir=source_path,
                use_labeled=use_labeled
            )

    def get_existing_collection(self):
        for collection in self.client.list_collections():
            if collection.name == self.db_name:
                self.collection = collection
                print(f"Collection {self.db_name} was found in database.")
                if self.get_item_state_baseline():
                    return True
                else:
                    print(f"Could not retrieve item-state baseline.")
                    return False
        print(f"No collection with name {self.db_name} was found in database.")
        return False
    
    def add_folder_to_collection(self, source_dir: str, use_labeled: bool=True):
        # input handling
        # check source path
        abs_source_path = os.path.abspath(source_dir)
        if not os.path.exists(abs_source_path) or not os.path.isdir(abs_source_path):
            print(f"Source path {abs_source_path} does not exist.")
            return
        
        # get collection
        if not self.get_existing_collection():
            print("No collection was found to add folder content to.")
            return
        
        # get room and precaution
        room = os.path.normpath(abs_source_path).split(os.path.sep)[-2]
        precaution = os.path.normpath(abs_source_path).split(os.path.sep)[-1]
        # get labeled images
        _source_path = abs_source_path
        if use_labeled:
            _source_path = os.path.join(_source_path, 'labeled')
        if not os.path.exists(_source_path) or not os.path.isdir(_source_path):
            print(f"Source directory {_source_path} does not exist.")
            return
        
        # get image files
        allowed_types = ['.jpeg', '.jpg', '.png']
        image_file_list = [os.path.join(_source_path, f) for f in os.listdir(_source_path) if os.path.splitext(f)[1].lower() in allowed_types and 'digital-' in f]
        if not image_file_list:
            print(f"No images were found in {_source_path}")
            return
        
        # get metadata file
        metadata_file = [os.path.join(abs_source_path, f) for f in os.listdir(abs_source_path) if os.path.splitext(f)[1].lower() in ['.csv'] and os.path.splitext(f)[0] == 'metadata']
        if not metadata_file:
            print(f"No metadata was found in {abs_source_path}")
            return
        metadata_file = metadata_file[0]

        # open checklist
        metadata = pd.read_csv(metadata_file)

        # metadata matching
        print(f"Creating database from {source_dir}")
        # get image indices from check list in first column
        metadata_list = metadata.iloc[:, 0]
        # try to match each image to a row in check list
        for image in image_file_list:
            file_name = os.path.basename(image)
            # get file number from image file
            file_no = re.search(r'\d+', file_name)
            if file_no:
                file_no = int(file_no[0])
                # iterate through check list to find matching image number
                for pd_idx, image_no in enumerate(metadata_list):
                    if image_no == image_no:
                        if int(image_no) == file_no:
                            # create metadata dict from base dict
                            metadata_dict = {}
                            for key, value in self.item_state_baseline.items():
                                # metadata_dict[key] = value['base_state']
                                metadata_dict[key] = 'n/a'
                            # get id from image name
                            id = f"{room}_{precaution}_{os.path.splitext(file_name)[0]}"
                            # get metadata from checklist
                            for item in metadata.iloc[pd_idx]:
                                if isinstance(item, str):
                                    data = item.lstrip(' ').rstrip(' ').split(': ')
                                    if len(data) == 2:
                                        key = data[0].replace(' ', '_')
                                        if key in metadata_dict.keys():
                                            metadata_dict[key] = data[1]
                                        else:
                                            print(f"Key '{key}' couldn't be found in {metadata_dict.keys()} for {image}.")
                                            metadata_dict[key] = 'n/a'
                            # add image and metadata to collection
                            metadata_dict['room'] = room
                            metadata_dict['precaution'] = precaution
                            self.collection.add(
                                documents=image,
                                metadatas=metadata_dict,
                                ids=id
                            )
                            # print(f"Metadata for ID {id} was added")

    
    def get_item_state_baseline(self):
        retrieved_doc = self.collection.get('item_state_baseline')
        # check if image was retrieved
        if not retrieved_doc['ids']:
            print(f"Item-state baseline was not found in collection.")
            return False
        self.item_state_baseline = json.loads(retrieved_doc['documents'][0])
        return True

    def get_image_by_id(self, image_id: str):
        retrieved_image = self.collection.get(image_id)
        # check if image was retrieved
        if not retrieved_image['ids']:
            print(f"No image was retrieved for {image_id}")
            return
        return retrieved_image
    
    def get_image_by_metadata(self, metadata: dict):
        search_array = []
        for key, value in metadata.items():
            search_array.append({key: {'$eq': value}})
        search_dict = {
            "$and": search_array
        }
        retrieved_data = self.collection.get(
            where=search_dict
        )
        if not retrieved_data['ids']:
            print(f"Data could not be retrieved for {metadata}")
            return
        if len(retrieved_data['ids']) > 1:
            print(retrieved_data)
            print("Multiple elements match search criteria. Check metadata")
            return retrieved_data
        return retrieved_data