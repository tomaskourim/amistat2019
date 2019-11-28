C_LAMBDAS = [0.01, 0.1, 0.25, 0.5, 0.8, 0.9, 0.99]
C_LAMBDA_PAIRS = [[0.1, 0.2], [0.2, 0.5], [0.5, 0.8], [0.1, 0.9], [0.5, 0.99], [0.8, 0.01], [0.99, 0.9]]
START_PROBABILITIES = [0.01, 0.1, 0.25, 0.5, 0.8, 0.9, 0.99]
STEP_COUNTS = [2, 3, 4, 5, 10, 50, 100]
REPETITIONS = 100

C_LAMBDAS_TESTING = [0.2, 0.9, 0.99]
C_LAMBDA_PAIRS_TESTING = [[0.5, 0.8], [0.99, 0.9], [0.2, 0.6]]
START_PROBABILITIES_TESTING = [0.99, 0.8, 0.4]
STEP_COUNTS_TESTING = [100, 1000]

WALK_TYPES = ['success_punished', 'success_rewarded', 'success_punished_two_lambdas', 'success_rewarded_two_lambdas']
DATA_DIRNAME = "generated_walks"

CONFIDENCE_INTERVAL_SIZE = 0.1
