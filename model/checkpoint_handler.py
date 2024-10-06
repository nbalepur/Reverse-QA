import pickle
import os

class Checkpoint:

    def __init__(self, args):
        self.results_dir = f'{args.res_dir}/{args.model_nickname}'
        self.partition = args.partition
        self.run_name = args.run_name

    def setup_partition(self, dataset_size):

        partition_map = {'full': (0, dataset_size),
        'first_half': (0, int(0.5 * dataset_size)),
        'second_half': (int(0.5 * dataset_size), dataset_size),
        'first_quarter': (0, int(0.25 * dataset_size)),
        'second_quarter': (int(0.25 * dataset_size), int(0.5 * dataset_size)),
        'third_quarter': (int(0.5 * dataset_size), int(0.75 * dataset_size)),
        'fourth_quarter': (int(0.75 * dataset_size), dataset_size),
        'first_eighth': (0, int(0.125 * dataset_size)),
        'second_eighth': (int(0.125 * dataset_size), int(2*0.125 * dataset_size)),
        'third_eighth': (int(2*0.125 * dataset_size), int(3*0.125 * dataset_size)),
        'fourth_eighth': (int(3*0.125 * dataset_size), int(4*0.125 * dataset_size)),
        'fifth_eighth': (int(4*0.125 * dataset_size), int(5*0.125 * dataset_size)),
        'sixth_eighth': (int(5*0.125 * dataset_size), int(6*0.125 * dataset_size)),
        'seventh_eighth': (int(6*0.125 * dataset_size), int(7*0.125 * dataset_size)),
        'eighth_eighth': (int(7*0.125 * dataset_size), dataset_size),
        }

        if self.partition not in partition_map:
            raise ValueError(f"The given partition is invalid: {self.partition}")
        start, end = partition_map[self.partition]
        self.start = start
        self.end = end

        return self.start, self.end

    def set_directories(self, pt):

        if self.partition == 'full':
            final_res_dir = f'{self.results_dir}/{self.run_name}/{pt.value}.pkl'
            final_res_dir_temp = f'{self.results_dir}/{self.run_name}/{pt.value}_temporary.pkl'
        else:
            final_res_dir = f'{self.results_dir}/{self.run_name}/{self.partition}/{pt.value}.pkl'
            final_res_dir_temp = f'{self.results_dir}/{self.run_name}/{self.partition}/{pt.value}_temporary.pkl'

        self.final_res_dir = final_res_dir
        self.final_res_dir_temp = final_res_dir_temp

    def load_checkpoint(self):

        if os.path.exists(self.final_res_dir):
            with open(self.final_res_dir, 'rb') as handle:
                outputs = pickle.load(handle)
                return outputs

        if not os.path.exists(self.final_res_dir_temp):
            return {'raw_text': [], 'prompt': []}
        
        with open(self.final_res_dir_temp, 'rb') as handle:
            outputs = pickle.load(handle)
        return outputs

    def save_checkpoint(self, outputs, is_final):

        out_dir = self.final_res_dir if is_final else self.final_res_dir_temp
        folder_path = '/'.join(out_dir.split('/')[:-1])
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with open(out_dir, 'wb') as handle:
            pickle.dump(outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def get_final_dir(self):
        return self.final_res_dir