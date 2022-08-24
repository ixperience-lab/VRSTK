import os

# Opening the file with absolute path
raw_data_file = open(r'MSSQAnswers.txt', 'rb')

mssq_Answers_file = 'MSSQAnswers_.txt'
mssq_Answers_file_content = ''

# read file
for line in raw_data_file:
    firstReplace = line.decode('utf-8', 'ignore').replace("\"\",\"\"", " # ")
    #print (firstReplace)
    secondReplace = firstReplace.replace(",\"\"", " # ")
    #print (secondReplace)
    thirdReplace = secondReplace.replace("\"\"", "")
    #print (thirdReplace)
    fourthReplace = thirdReplace.replace("\"", "")
    #print (fourthReplace)
    fifthReplace = fourthReplace.replace(";", ",")
    #print (fifthReplace)
    mssq_Answers_file_content += fifthReplace
    
# Closing the file after reading
raw_data_file.close()

# ".\\"
#-------------------------------
if os.path.exists(mssq_Answers_file):
    os.remove(mssq_Answers_file)

# open file for writting
mssq_Answers_file_open = open(mssq_Answers_file, "w")
# write to file
mssq_Answers_file_open.write(mssq_Answers_file_content)
# Closing the file after writting
mssq_Answers_file_open.close()