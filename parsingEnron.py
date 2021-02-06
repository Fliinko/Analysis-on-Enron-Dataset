import os
from email.parser import Parser

#Object Email
class Email:
    def __init__(self, sender, recipients, body):
        self.sender = sender
        self.recipients = recipients
        self.body = body

    def __str__(self):
        return "From: %s \nTo: %s \n\n%s" %(self.sender, self.recipients, self.body)

rootdir = "maildirtest"

for directory, subdirectory, filenames in  os.walk(rootdir):
    print(directory, subdirectory, len(filenames))

emails = []

for directory, subdirectory, filenames in  os.walk(rootdir):
    for filename in filenames:

        #Reading data from file
        with open(os.path.join(directory, filename), "r") as f:
            data = f.read()

        #Creating instance of the email.parser object
        emailParser = Parser().parsestr(data)

        #reading the from section of the email
        sender = emailParser['from']

        #reading the to section of the email, which can contain multiple recipients
        if emailParser['to']:
            recipients = emailParser['to']
            recipients = "".join(recipients.split())
            recipients = recipients.split(",")
        else:
            recipients = ['None']

        #reading the body section of the email
        body = emailParser.get_payload()

        #Creating an email object and appending it to the list of all emails
        email = Email(sender, recipients, body)
        emails.append(email)

#print(emails[2])
#for email in emails:
#print(email.recipients)

dataset = {}

for email in emails:
    for recipient in email.recipients:
        key1 = (email.sender, recipient)
        key2 = (recipient, email.sender)
        if(key1 in dataset.keys()):
            dataset[email.sender, recipient] += email.body
        elif (key2 in dataset.keys()):
            dataset[recipient, email.sender] += email.body
        else:
            dataset[email.sender, recipient] = email.body

#counter = 0
#for data in dataset:
#    counter += 1
#    print("\n\n\n\n\n")
#    print(dataset[data])
#    if(counter == 2):
#        break

