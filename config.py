C_LAMBDAS = [0.5, 0.8, 0.9, 0.99]
C_LAMBDA_PAIRS = [[0.5, 0.8], [0.1, 0.5], [0.5, 0.99], [0.99, 0.9]]
START_PROBABILITIES = [0.5, 0.8, 0.9, 0.99]
STEP_COUNTS = [5, 10, 50, 100]
REPETITIONS_OF_WALK = 100
REPETITIONS_OF_WALK_SERIES = 100

C_LAMBDAS_TESTING = [0.2, 0.75, 0.95]
C_LAMBDA_PAIRS_TESTING = [[0.5, 0.8], [0.75, 0.9], [0.2, 0.6]]
START_PROBABILITIES_TESTING = [0.99, 0.8, 0.4]
STEP_COUNTS_TESTING = [100, 1000]

MODEL_TYPES = ['success_punished', 'success_rewarded', 'success_punished_two_lambdas', 'success_rewarded_two_lambdas']
DATA_DIRNAME = "generated_walks"
PREDICTION_TYPES = ["only_lambda", "only_p0", "all_parameters", "everything"]

OPTIMIZATION_ALGORITHM = 'Nelder-Mead'

CONFIDENCE_INTERVAL_SIZE = 0.1

MODEL_PARAMETERS = ["model_type", "c_lambda", "c_lambda0", "c_lambda1", "p0", "step_count"]
PREDICTION_VALUES = ["predicted_model", "predicted_lambda", "predicted_lambda0", "predicted_lambda1", "predicted_p0"]
