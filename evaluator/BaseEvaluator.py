import os
import yaml

class BaseEvaluator():
    def __init__(self, name) -> None:
        self.name = name
        if not os.path.exists('evals/' + self.name):
            os.makedirs('evals/' + self.name, exist_ok=True)

        self.eval_directory = self.create_new_evaluate_directory()

    def create_new_evaluate_directory(self):
        for i in range(1000):
            if not os.path.exists('evals/' + self.name + '/eval_' + str(i)):
                self.run_id = i
                os.makedirs('evals/' + self.name + '/eval_' + str(i), exist_ok=True)
                break
        return 'evals/' + self.name + '/eval_' + str(self.run_id)

    def get_eval_directory(self):
        return self.eval_directory
    
    def get_params_path(self):
        return self.eval_directory + '/' + 'params.yaml'
    
    def save_evaluate_params(self, class_name):
        class_attributes = class_name.__dict__

        # Remove special methods and class attributes
        class_attributes = {key: value for key, value in class_attributes.items() if not key.startswith("__") and not callable(value)}

        # Save class attributes to YAML file
        with open(self.get_params_path(), 'w') as file:
            yaml.dump(class_attributes, file, default_flow_style=False)
        