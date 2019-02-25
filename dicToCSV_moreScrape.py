import csv

csv_columns = ['agency','title','article']
dict_data = [
{'agency': 'BBC', 'title': 'T1', 'article': 'A1'},
{'agency': 'BBC', 'title': 'T2', 'article': 'A2'},
{'agency': 'BBC', 'title': 'T3', 'article': 'A3'},
{'agency': 'BBC', 'title': 'T4', 'article': 'A4'},
{'agency': 'BBC', 'title': 'T5', 'article': 'A5'},
]

dict_check = [
{'agency': 'BBC', 'title': 'T1', 'article': 'A1_ceck'},
{'agency': 'BBC', 'title': 'T2', 'article': 'A2_check'},
{'agency': 'NYT', 'title': 'T6', 'article': 'A3'},
{'agency': 'NYT', 'title': 'T4', 'article': 'A4'},
{'agency': 'NYT', 'title': 'T5', 'article': 'A5'},
]

csv_file = 'real_or_opinion.csv'

# initiates csv file and writes headers
# use when running script for first time 
# i.e no existing data in csv file
try:
    with open(csv_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
except IOError:
    print("I/O error") 

# checks for repeated articles
agency_list = []
title_list = []
try:
    with open(csv_file) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            agency_list = agency_list + [row['agency']]
            title_list = title_list + [row['title']]
except IOError:
    print("I/O error") 

#print(title_list)

# appends data to csv file
try:
    with open(csv_file, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        for data in dict_data:
            if data['title'] not in title_list:
                writer.writerow(data)
            else:
                indices = [i for i, x in enumerate(title_list) if x == data['title']]   # gets list of indexes where title is the same
                agencies = [agency_list[index] for index in indices]             # selects the correspoding agencies
                for agency in agencies:
                    # checks if agency is the same 
                    if data['agency'] != agency:
                        writer.writerow(data)
except IOError:
    print("I/O error") 