import torch
import numpy as np
import base64
import csv
import os
import time
import joblib
from collections import defaultdict
from sklearnex import patch_sklearn
patch_sklearn(['KMeans','DBSCAN'])
from sklearn.cluster import KMeans

class LoadZdict():
    def __init__(self, img_zdict_file, obj_zdict_file, txt_zdict_file):
        self.obj_tsv_fieldnames = ['feature', 'bbox', 'heading', 'elevation', 'clsprob', 'clsname', 'pz']
        self.img_tsv_fieldnames = ['roomtype','feature','pz']
        self.txt_tsv_fieldnames = ['token_type','token','feature','pz']
        self.img_zdict_file = img_zdict_file
        self.obj_zdict_file = obj_zdict_file
        self.txt_zdict_file = txt_zdict_file
    
    def read_obj_tsv(self):
        in_data = []
        with open(self.obj_zdict_file, 'rt') as tsv_in_file:
            reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = self.obj_tsv_fieldnames)
            for item in reader:
                item['feature'] = np.frombuffer(base64.b64decode(item['feature']), dtype=np.float32)
                item['bbox'] = np.frombuffer(base64.b64decode(item['bbox']), dtype=np.float32)
                item['heading'] = float(item['heading'])
                item['elevation'] = float(item['elevation'])
                item['clsprob'] = np.frombuffer(base64.b64decode(item['clsprob']), dtype=np.float32)
                item['pz'] = float(item['pz'])
                in_data.append(item)
        return in_data

    def read_img_tsv(self):
        in_data = []
        with open(self.img_zdict_file, 'rt') as tsv_in_file:
            reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = self.obj_tsv_fieldnames)
            for item in reader:
                item['feature'] = np.frombuffer(base64.b64decode(item['feature']), dtype=np.float32)
                item['pz'] = float(item['pz'])
                in_data.append(item)
        return in_data

    def read_instr_tsv(self):
        in_data = []
        with open(self.txt_zdict_file, 'rt') as tsv_in_file:
            reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = self.txt_tsv_fieldnames)
            for item in reader:
                item['feature'] = np.frombuffer(base64.b64decode(item['feature']), dtype=np.float32)
                item['pz'] = float(item['pz'])
                in_data.append(item)
        return in_data
    
    def load_all_zdicts(self):
        obj_zdict, img_zdict, instr_zdict = self.read_obj_tsv(), self.read_img_tsv(), self.read_instr_tsv()
        return obj_zdict, img_zdict, instr_zdict

    def load_obj_tensor(self,is_random=False):
        ''' return corresponding tensors
        '''
        obj_features = []
        obj_locs = []
        obj_clsprobs = []
        obj_pzs = []
        
        obj_available_index = [] # record available index (after filtering)
        with open(self.obj_zdict_file, 'rt') as tsv_in_file:
            reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = self.obj_tsv_fieldnames)
            for item in reader:
                obj_feature = np.frombuffer(base64.b64decode(item['feature']), dtype=np.float32)
                if is_random:
                    obj_features.append(np.random.random(obj_feature.shape).astype(np.float32))
                else:
                    obj_features.append(obj_feature)
                # obj_features.append(np.frombuffer(base64.b64decode(item['feature']), dtype=np.float32))
                obj_bbox = np.frombuffer(base64.b64decode(item['bbox']), dtype=np.float32)
                obj_heading = float(item['heading'])
                obj_elevation = float(item['elevation'])
                obj_loc = np.array([obj_bbox[0],
                                    obj_bbox[1],
                                    obj_bbox[2],
                                    obj_bbox[3],
                                    obj_bbox[4],
                                    np.sin(obj_heading),np.cos(obj_heading),np.sin(obj_elevation),np.cos(obj_elevation)])
                obj_locs.append(obj_loc)
                obj_clsprob = np.frombuffer(base64.b64decode(item['clsprob']), dtype=np.float32)
                obj_clsprobs.append(obj_clsprob)
                obj_pzs.append(float(item['pz']))

                obj_index = np.argmax(obj_clsprob)
                if obj_index not in obj_available_index:
                    obj_available_index.append(obj_index)
        return {
            "obj_features": torch.from_numpy(np.array(obj_features)).cuda(),
            "obj_locs": torch.from_numpy(np.array(obj_locs)).cuda(),
            "obj_clsprobs": torch.from_numpy(np.array(obj_clsprobs)).cuda(),
            "obj_pzs": torch.from_numpy(np.array(obj_pzs)).cuda(),
            "obj_indexs": obj_available_index
        }
    
    def load_img_tensor(self,is_random=False):
        img_features = []
        img_pzs = []
        with open(self.img_zdict_file, 'rt') as tsv_in_file:
            reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = self.img_tsv_fieldnames)
            for item in reader:
                img_feature = np.frombuffer(base64.b64decode(item['feature']), dtype=np.float32)
                if is_random:
                    img_features.append(np.random.random(img_feature.shape).astype(np.float32))
                else:
                    img_features.append(img_feature)
                # img_features.append(np.frombuffer(base64.b64decode(item['feature']), dtype=np.float32))
                img_pzs.append(float(item['pz']))
        return {
            "img_features": torch.from_numpy(np.array(img_features)).cuda(),
            "img_pzs": torch.from_numpy(np.array(img_pzs)).cuda()
        }

    def load_instr_tensor(self, is_random=False):
        instr_direction_features = []
        instr_direction_pzs = []
        instr_landmark_features = []
        instr_landmark_pzs = []
        with open(self.txt_zdict_file, 'rt') as tsv_in_file:
            reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = self.txt_tsv_fieldnames)
            for item in reader:
                if item['token_type'] == 'direction':
                    feature = np.frombuffer(base64.b64decode(item['feature']), dtype=np.float32)
                    if is_random:
                        instr_direction_features.append(np.random.random(feature.shape).astype(np.float32))
                    else:
                        instr_direction_features.append(feature)
                    # instr_direction_features.append(np.frombuffer(base64.b64decode(item['feature']), dtype=np.float32))
                    instr_direction_pzs.append(float(item['pz']))
                elif item['token_type'] == 'landmark':
                    feature = np.frombuffer(base64.b64decode(item['feature']), dtype=np.float32)
                    if is_random:
                        instr_landmark_features.append(np.random.random(feature.shape).astype(np.float32))
                    else:
                        instr_landmark_features.append(feature)
                    # instr_landmark_features.append(np.frombuffer(base64.b64decode(item['feature']), dtype=np.float32))
                    instr_landmark_pzs.append(float(item['pz']))
        if len(instr_direction_features) != 0:
            return {
                "instr_direction_features": torch.from_numpy(np.array(instr_direction_features)).cuda(),
                "instr_direction_pzs": torch.from_numpy(np.array(instr_direction_pzs)).cuda(),
                "instr_landmark_features": torch.from_numpy(np.array(instr_landmark_features)).cuda(),
                "instr_landmark_pzs": torch.from_numpy(np.array(instr_landmark_pzs)).cuda(),
            }
        else: # reverie
            return {
                "instr_landmark_features": torch.from_numpy(np.array(instr_landmark_features)).cuda(),
                "instr_landmark_pzs": torch.from_numpy(np.array(instr_landmark_pzs)).cuda(),
            }

# ==========
# Use KMeans to randomly pick features
# ==========
class KMeansPicker():
    def __init__(self, front_feat_file, kmeans_file=None, n_clusters=256):
        self.TIM_TSV_FIELDNAMES = ['path_id', 'txt_feats', 'vp_feats', 'gmap_feats']
        self.n_clusters = n_clusters
        txt_feats, vp_feats, gmap_feats = self.read_tim_tsv(front_feat_file)
        self.feat_dicts = {
            'txt_feats': txt_feats,
            'vp_feats': vp_feats,
            'gmap_feats': gmap_feats
        }
        if kmeans_file is not None:
            self.kmeans_model_dict = {
                'txt_feats': joblib.load(os.path.join(kmeans_file,'txt_feats.pkl')),
                'vp_feats': joblib.load(os.path.join(kmeans_file,'vp_feats.pkl')),
                'gmap_feats': joblib.load(os.path.join(kmeans_file,'gmap_feats.pkl')),
            }
        else:
            self.kmeans_model_dict = {}
            for k, v in self.feat_dicts.items():
                kmeans = KMeans(n_clusters=n_clusters)
                start_time = time.time()
                kmeans.fit(v)
                print('Finish KMeans on %d %s. The total time is %.2f min.' %(n_clusters, k, (time.time()-start_time)/60))

                self.kmeans_model_dict[k] = kmeans

        print('Successfully load front features and KMeans models.')

    def read_tim_tsv(self, front_feat_file, return_dict=False):
        txt_feats = []
        vp_feats = []
        gmap_feats = []
        with open(front_feat_file, 'rt') as tsv_in_file:
            reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = self.TIM_TSV_FIELDNAMES)
            for item in reader:
                txt_feats.append(np.frombuffer(base64.b64decode(item['txt_feats']), dtype=np.float32))
                vp_feats.append(np.frombuffer(base64.b64decode(item['vp_feats']), dtype=np.float32))
                gmap_feats.append(np.frombuffer(base64.b64decode(item['gmap_feats']), dtype=np.float32))
        if return_dict:
            feat_dict = {
                'txt_feats': txt_feats,
                'vp_feats': vp_feats,
                'gmap_feats': gmap_feats
            }
            return feat_dict
        else:
            return np.array(txt_feats), np.array(vp_feats), np.array(gmap_feats)

    def random_pick_front_features(self, z_front_log_dir, iter, save_file=False, define_value=''):
        '''define_value: to verify loading random values
        '''
        random_feat_dicts = defaultdict(lambda: [])
        for k in self.feat_dicts.keys():
            kmeans = self.kmeans_model_dict[k]
            # Iterate over the unique cluster labels
            for cluster_label in np.unique(kmeans.labels_):
                # Find the indices of samples belonging to the current cluster
                cluster_indices = np.where(kmeans.labels_ == cluster_label)[0]
                
                # Randomly select one sample from the current cluster
                random_sample_index = np.random.choice(cluster_indices)
                random_sample = self.feat_dicts[k][random_sample_index]
                if define_value=='random':
                    new_random_sample = np.random.rand(*random_sample.shape).astype(random_sample.dtype)
                    random_sample = new_random_sample
                elif define_value=='zero':
                    new_random_sample = np.zeros(*random_sample.shape).astype(random_sample.dtype)
                    random_sample = new_random_sample
                
                # Add the randomly picked sample to the list
                random_feat_dicts[k].append(random_sample)
        if save_file:
            # Save front features for inference
            target_file = os.path.join(z_front_log_dir, f'z_front_feature_{iter}.tsv')
            with open(target_file, 'wt') as tsvfile:
                writer = csv.DictWriter(tsvfile, delimiter = '\t', fieldnames = self.TIM_TSV_FIELDNAMES)
                for i in range(self.n_clusters):
                    record = {
                        'path_id': 0,
                        'txt_feats': str(base64.b64encode(random_feat_dicts['txt_feats'][i]), "utf-8"),
                        'vp_feats': str(base64.b64encode(random_feat_dicts['vp_feats'][i]), "utf-8"),
                        'gmap_feats': str(base64.b64encode(random_feat_dicts['gmap_feats'][i]), "utf-8"),
                    }
                    writer.writerow(record)
        return random_feat_dicts