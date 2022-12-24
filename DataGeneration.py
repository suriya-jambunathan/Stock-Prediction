# Import modules for data manipulation and visualization
import pandas as pd
import numpy as np
import yfinance as yf

# Import modules for image processing
from PIL import Image

# Import module for progress bar
from tqdm import tqdm

# Import module for mathematical functions
import math

# Import module for interacting with the operating system
import os

# Import module for plotting
import matplotlib.pyplot as plt

class DataGeneration():

    def __init__(self, base_path = 'Dataset/'):
        """
        Initializes the class with the base path of the directory where the dataset will be stored.
        The default value of the base path is 'Dataset/'

        Parameters:
        base_path (str): The base path of the directory where the dataset will be stored.
        """
        self.pixels = (150, 150)
        self.dpi = 100
        self.base_path = base_path

    def generate_image_data(self, comp, period):
        """
        Generates the image data for the given company's stock prices for a given period.
        
        Parameters:
        comp (str): The name of the company for which the image data is to be generated.
        period (int): The period for which the image data is to be generated.
        """
        # Set image properties
        pixels = self.pixels
        dpi = self.dpi

        # Download data for the given stock symbol
        df = yf.download(
                        tickers=[comp], 
                        period="max",
                        interval="1d",
                        progress = False)

        # Extract relevant information from the data
        period_list = []
        date_ranges = []
        perc_changes = []
        adj_close = list(df['Adj Close'])
        dates = list(df.index.strftime('%d%b%Y'))
        for i in range(len(df) - period):
            period_list.append(adj_close[i:i+period])
            date_ranges.append(f"{dates[i]}_{dates[i+period]}")
        for i in range(len(period_list)-1):
            chg = (period_list[i+1][0] - period_list[i][-1])*100/period_list[i][-1]
            if chg >= 0:
                perc_changes.append(math.floor(chg))
            else:
                perc_changes.append(math.ceil(chg))

        period_list = period_list[:-1]
        date_ranges = date_ranges[:-1]

        # Create a folder for the stock symbol
        os.mkdir(comp + '/')

        # Generate and save image for each period of data
        plt.ioff()
        for i in range(len(period_list)):
            plt.figure(figsize = (pixels[0]/dpi, pixels[1]/dpi), dpi = dpi)
            plt.style.use('dark_background')
            plt.axis('off')
            plt.plot(period_list[i], color = 'white')
            file_name = f"{comp}/{comp}__{date_ranges[i]}__{period}period__{perc_changes[i]}.png"
            plt.savefig(file_name, dpi = 100)
    
    def generate_csv(self):
        """
        Generates a CSV file for each subdirectory in the base directory.
        The CSV file contains information about images in the subdirectory, including the image data,
        company name, period, date range, and percentage change.

        Parameters:
            self: instance of the current object
        """
        # Get the base path and list of subdirectories in the base path
        base_path = self.base_path
        sub_dirs = [os.path.join(base_path, name) for name in os.listdir(base_path) 
                                                        if os.path.isdir(os.path.join(base_path, name))]

        datasets = []
        # Iterate through each subdirectory
        for sub_dir in sub_dirs:
            # Get a list of all files in the subdirectory, including those in subdirectories within the subdirectory
            files_list = []
            for root, directories, files in os.walk(sub_dir + '/'):
                for name in files:
                    files_list.append(os.path.join(root, name))
            # Only keep files with the '.png' extension
            files_list = [file for file in files_list if '.png' in file]

            # Initialize lists to store image data, company names, periods, date ranges, and percentage changes
            images = []
            comps = []
            periods = []
            date_ranges = []
            perc_changes = []

            # Iterate through each file in the list
            for file in tqdm(files_list): 
                # Load the image and convert it to grayscale, resize it to (16, 16), and flatten it
                image = np.asarray(Image.open(file).convert("L").resize((16, 16))).flatten()
                # Normalize the image data and add it to the list
                images.append(list(2*(image/255 - 0.5)))
                # Extract the company name, period, date range, and percentage change from the file name
                fn = file.split('/')[-1].split('__')
                comps.append(fn[0])
                periods.append(int(fn[2].split('period')[0]))
                date_ranges.append(fn[1])
                perc_changes.append(int(fn[-1].split('.')[0]))

            # Create a Pandas dataframe with the image data, company names, periods, date ranges, and percentage changes
            dataset = pd.DataFrame(images)
            dataset['Company'] = comps
            dataset['Period'] = periods
            dataset['Date Range'] = date_ranges
            dataset['Change'] = perc_changes

            # Generate a file name for the CSV file based on the company name, period, and number of samples
            file_name = f"{base_path}{dataset['Company'][0]}__{dataset['Period'][0]}Period__{len(dataset)}Samples.csv"
            # Save the dataframe as a CSV file
            dataset.to_csv(file_name)
            datasets.append(dataset)
        
        #Generating cummulative dataset
        overall_dataset = pd.concat(datasets)
        overall_dataset = overall_dataset.reset_index(drop = True)
        overall_dataset.to_csv('Dataset.csv')