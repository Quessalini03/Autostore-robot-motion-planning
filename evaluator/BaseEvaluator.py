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