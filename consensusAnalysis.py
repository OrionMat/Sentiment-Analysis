from bs4 import BeautifulSoup
from googlesearch import search 

query = "Geeksforgeeks"

for j in search(query, tld="co.in", num=10, stop=10, pause=2): 
	print(j) 

print('\n')

query = "A computer science portal"
  
for j in search(query, tld="co.in", num=10, stop=1, pause=2): 
    print(j) 