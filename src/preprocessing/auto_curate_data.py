# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 16:00:28 2023

@author: SanjayanA
"""

import pandas as pd
import dask.dataframe as dd
import numpy as np
import ast
import os
import brpylib
import traceback


def generate_timestamps(start_time_micro, length, sfreq):
    """
    Generate timestamps given start time, length of the signal, and sampling frequency.
    """
    # Calculate the end time based on the length and sampling frequency
    end_time_micro = start_time_micro + ((length - 1) / sfreq)*10**6

    # Generate timestamps
    timestamps = np.linspace(start_time_micro, end_time_micro, length)
    
    return timestamps.astype(np.int64)



base_dir = r'O:\PINT\_Projects\Bladder\012_19 SPARC\Data'

manifest = pd.read_csv('manifest.csv')

existing_files = os.listdir()
exempt = ['manifest.csv', 'VoidInfo.csv' , 'auto.py']
for e in exempt:
    if (e in existing_files):
        existing_files.remove(e)

record_types = ['saline']

for index, row in manifest.iterrows():
    subject = row['subject']
    testType = row['type']
    for r_t in record_types:
        try:
            record_spec = ast.literal_eval(row[r_t])
        except:
            raise Exception(f"Check {subject} | {r_t} entry. Not valid!")
        if (type(record_spec) is not dict):
            raise Exception(f"Check {subject} | {r_t} entry. Not valid dict!")
        
        try:
            for day in record_spec.keys():
                for record in record_spec[day]:
                    print(f"Subject {subject}, Day {day}, record {record}")
                    
                    file_name = f"{subject}.{testType}.{r_t}.{day}.{record}.parquet"
                    if (file_name in existing_files):
                        existing_files.remove(file_name)
                        continue
                    
                    # File doesn't exist already, so try to create it
                    
                    # navigate to the corresponding subject and day folder
                    day_dir = os.path.join(base_dir, subject, f'Day {day}')
                    
                    # verify the directory exists
                    if not os.path.isdir(day_dir):
                        print(f"No directory found for {day_dir}")
                        continue
                    
                    # loop through all the files in the day directory
                    found_files = {}
                    req_formats = ['.ns2', '.ns5']
                    for file in os.listdir(day_dir):
                        # check if the file is an .ns2 or .ns5 file and if its last three characters correspond to record
                        try:
                            if file[-4:] in req_formats and int(file[-7:-4]) == record:
                                found_files[file[-4:]] = os.path.join(day_dir, file)
                                print(f"File {file} found for Subject {subject}, Day {day}, Record {record}")
                        except:
                            print("invalid file")
                    
                    if all(extension in found_files for extension in req_formats):
                        # Found specified files
                        print(found_files)
                        cont_data = {}
                        for fk in found_files.keys():
                            nsx_file = found_files[fk]
                            open_file = brpylib.NsxFile(nsx_file)
                            cont_data[fk] = open_file.getdata('all')
                            open_file.close()
                        
                        seg_id = 0
                        
                        ch_pressure = cont_data['.ns2']["elec_ids"].index(133)
                        ch_spine    = cont_data['.ns5']["elec_ids"].index(135)
                        ch_bladder  = cont_data['.ns5']["elec_ids"].index(136)
                        
                        
                        start_time = os.path.getctime(found_files['.ns2'])
                        
                        
                        rawPressureData = cont_data['.ns2']["data"][seg_id][ch_pressure]
                        pressure_sfreq = cont_data['.ns2']["samp_per_s"]
                        
                        rawSpineData = cont_data['.ns5']["data"][seg_id][ch_spine]
                        nSpine_sfreq = cont_data['.ns5']["samp_per_s"]
                        
                        rawBladderData = cont_data['.ns5']["data"][seg_id][ch_bladder]
                        nBladder_sfreq = cont_data['.ns5']["samp_per_s"]
                        
                        scaleFactor = 0.000152592547379986
                        
                        k_PressureOffset = row['k_PressureOffset']
                        k_PressureCalibration = row['k_PressureCalibration']
                        
                        Pressure = ((rawPressureData*scaleFactor - k_PressureOffset)/k_PressureCalibration)
                        nSpine = (rawSpineData*scaleFactor)
                        nBladder = (rawBladderData*scaleFactor)
                        
                        # Convert start_time to microseconds
                        start_time_micro = start_time * 10**6
                        
                        # Generate time stamps for each data type
                        pressure_time_stamps = generate_timestamps(start_time_micro, len(Pressure), pressure_sfreq)
                        neural_time_stamps = generate_timestamps(start_time_micro, len(nSpine), nSpine_sfreq)
                        #nBladder_time_stamps = generate_timestamps(start_time_micro, len(nBladder), nBladder_sfreq)
                        
                        # Create dataframes
                        pressure_df = pd.DataFrame({
                            'time': pressure_time_stamps.astype(np.int64),
                            'Pressure': Pressure
                        }).set_index('time')
                        
                        neural_df = pd.DataFrame({
                            'time': neural_time_stamps.astype(np.int64),
                            'nSpine': nSpine,
                            'nBladder': nBladder
                        }).set_index('time')
                        
                        
                        # Convert to dask for concatenation (more efficient for large datasets)
                        dd_pressure = dd.from_pandas(pressure_df, npartitions=10)
                        dd_neural = dd.from_pandas(neural_df, npartitions=10)
                        
                        # Merge dataframes
                        ddf = dd.concat([dd_pressure,dd_neural], axis=1)
                        
                        # Convert back to pandas
                        df = ddf.compute()
                        
                        # Sort time values
                        df = df.sort_index()
                        
                        # Forward fill to handle blank data spaces
                        df.ffill(inplace=True)
                        
                        
                        # Save df
                        print(f"Saving as {file_name}")
                        
                        df.to_parquet(file_name)

        except:
            raise Exception(f"Check {subject} | {r_t} entry. Dict not formatted correctly")
            traceback.print_exc()

            
for rm_file in existing_files:
    print(f"Removing {rm_file}")
    os.remove(rm_file)