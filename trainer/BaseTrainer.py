import os
import yaml

class BaseTrainer():
    def __init__(self, name) -> None:
        self.name = name
        if not os.path.exists('runs/' + self.name):
            os.mkdir('runs/' + self.name)

        self.run_directory = self.create_new_run_directory()

    def create_new_run_directory(self):
        for i in range(1000):
            if not os.path.exists('runs/' + self.name + '/run_' + str(i)):
                self.run_id = i
                os.mkdir('runs/' + self.name + '/run_' + str(i))
                break
        return 'runs/' + self.name + '/run_' + str(self.run_id)

    def get_run_directory(self):
        return self.run_directory
    
    def get_model_path(self):
        return self.run_directory + '/' + 'model.pt'

    def get_hparams_path(self):
        return self.run_directory + '/' + 'hparams.yaml'
    
    def save_hyperparameters(self, class_name):
        class_attributes = class_name.__dict__

        # Remove special methods and class attributes
        class_attributes = {key: value for key, value in class_attributes.items() if not key.startswith("__") and not callable(value)}

        # Save class attributes to YAML file
        with open(self.get_hparams_path(), 'w') as file:
            yaml.dump(class_attributes, file, default_flow_style=False)