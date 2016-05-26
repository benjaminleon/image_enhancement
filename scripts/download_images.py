# Run this script from a folder so that the text file with urls are in the path
import urllib
import socket
import random
import time

def print_done( start_time, number_of_timeouts ):
    elapsed_time = time.time() - start_time
    if elapsed_time < 60:
        unit = " seconds"
    elif elapsed_time < 3600:
        elapsed_time = float(elapsed_time)/60
        unit = " minutes"
    else:
        elapsed_time = float(elapsed_time)/3600
        unit = " hours"
    elapsed_time = round(elapsed_time)
    print "Done! Collected " + str(number_of_samples - number_of_timeouts) + " of " + str(number_of_samples) + ", it took " + str(elapsed_time) + unit 
    print "Number of timeouts: " + str(number_of_timeouts)

filename = "fall11_urls.txt"
number_of_samples = 5

socket.setdefaulttimeout(20)
with open(filename) as urlfile:
    for number_of_urls, l in enumerate(urlfile): # Count the number of urls
        pass
    print "Number of urls: " + str(number_of_urls)

    random_lines = random.sample(range(0, number_of_urls), number_of_samples) # Select a random sample of urls
    random_lines = sorted(random_lines)
    
number_of_timeouts = 0
current_url_idx = 0    
url_number = random_lines[current_url_idx]
with open(filename) as urlfile: # Need to open it again to enumerate once more
    start_time = time.time()
    for i, line in enumerate(urlfile):
        if i == url_number:

            url_number = random_lines[current_url_idx]
                                    
            urlstart = line.find("http://")
            url = line[urlstart:]
            try:
                urllib.urlretrieve(url, "/home/ben/exjobb/test" + str(url_number) + ".jpg")
                print "Downloading " + str(1 + current_url_idx) + " of " + str(number_of_samples)
            except:
                print sys.exc_info()
                number_of_timeouts += 1
                
            current_url_idx += 1
            if current_url_idx >= number_of_samples: # We're done
                end_time = time.time()
                print_done(start_time, number_of_timeouts)
                break
            url_number = random_lines[current_url_idx]
            

                
    
