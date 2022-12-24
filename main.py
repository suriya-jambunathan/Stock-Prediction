# Import the DataGeneration and Test classes from the DataGeneration and Test modules, respectively
from DataGeneration import DataGeneration
from Test import Test

# Only run the following code if this script is the main module being run
if __name__ == '__main__':
    # Create an instance of the DataGeneration class with the base path parameter set to 'Dataset/'
    DataGenerator = DataGeneration(base_path='Dataset/')
    
    # Define a list of company names
    comps = [ 'PFE', 'NFLX', 'GOOG', 'AMD', 'CSCO', 'MSFT', 'BABA', 'DIS', 
              'TCEHY', 'NVDA', 'CMCSA', 'TM', 'AAPL', 'HMC', 'VSAT', 'QCOM',   
              'INTC', 'NDAQ', 'BMWYY', 'TSLA',  'NSANY', 'AMZN',  'COST', 'META']
    period = 10
    
    # Call the generate_image_data method on each company name in the comps list with the period parameter
    for comp in comps:
        DataGenerator.generate_image_data(comp, period)
    
    # Call the generate_csv method on the DataGeneration object
    DataGenerator.generate_csv()
    
    # Create an instance of the Test class
    test_obj = Test()

    # Run machine learning tests with no sampling
    # Create an instance of the ML class with the mode parameter set to 'binary'
    ml_no_sampling_obj = test_obj.ML(mode='binary')
    ml_no_sampling_obj.read_data('Dataset.csv')
    no_sampling_ml_content = ml_no_sampling_obj.evaluate_data()

    # Run machine learning tests with under-sampling
    # Create an instance of the ML class with the mode parameter set to 'binary' and the sampling parameter set to 'under'
    ml_under_sampling_obj = test_obj.ML(mode='binary', sampling='under')    
    ml_under_sampling_obj.read_data('Dataset.csv')
    under_sampling_ml_content = ml_under_sampling_obj.evaluate_data()

    # Run machine learning tests with over-sampling
    # Create an instance of the ML class with the mode parameter set to 'binary' and the sampling parameter set to 'over'
    ml_over_sampling_obj = test_obj.ML(mode='binary', sampling='over')    
    ml_over_sampling_obj.read_data('Dataset.csv')
    over_sampling_ml_content = ml_over_sampling_obj.evaluate_data()

    # Run multi-class machine learning tests with no sampling
    # Create an instance of the ML class with the mode parameter set to 'multi'
    multi_ml_no_sampling_obj = test_obj.ML(mode='multi')
    multi_ml_no_sampling_obj.read_data('Dataset.csv')
    multi_no_sampling_ml_content = multi_ml_no_sampling_obj.evaluate_data()

    # Create an instance of the ML class with the mode parameter set to 'multi' and the sampling parameter set to 'under'
    multi_ml_under_sampling_obj = test_obj.ML(mode='multi', sampling='under')
    multi_ml_under_sampling_obj.read_data('Dataset.csv')
    multi_under_sampling_ml_content = multi_ml_under_sampling_obj.evaluate_data()

    # Create an instance of the ML class with the mode parameter set to 'multi' and the sampling parameter set to 'over'
    multi_ml_over_sampling_obj = test_obj.ML(mode='multi', sampling='over')
    multi_ml_over_sampling_obj.read_data('Dataset.csv')
    multi_over_sampling_ml_content = multi_ml_over_sampling_obj.evaluate_data()

    # Run neural network tests in binary mode
    # Create an instance of the NN class with the mode parameter set to 'binary'
    nn_obj = test_obj.NN(mode='binary')
    nn_obj.read_data('Dataset.csv')
    nn_binary_content = nn_obj.evaluate_data()

    # Create an instance of the NN class with the mode parameter set to 'multi'
    multi_nn_obj = test_obj.NN(mode='multi')
    multi_nn_obj.read_data('Dataset.csv')
    nn_multi_content = multi_nn_obj.evaluate_data()