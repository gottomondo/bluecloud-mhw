import os,requests

#ccpiamurl=os.environ["ccpiamurl"]
#ccpclientid=os.environ["ccpclientid"]
#ccprefreshtoken=os.environ["ccprefreshtoken"]
#ccpcontext=os.environ["ccpcontext"]

print("Getting token")

#headers = { "X-D4Science-Context": ccpcontext }
#params = { "grant_type":"refresh_token", "client_id":ccpclientid, "refresh_token":ccprefreshtoken }
    
#response = requests.post(ccpiamurl, headers=headers, data=params).json()
#TOKEN = response["access_token"]

#print("Uploading")

#dev = "8f3d78fd-5c60-492f-9722-86a73a82c303"
#url = "https://api.d4science.org/workspace/items/"+dev+"/create/FILE"

#headers = { "Authorization": "Bearer " + TOKEN }

#files = { "name": "Figure_2", "description":"Figure generated by the Maps Python script", "file": (open('/output/Figure_2.png', 'rb')) }

#response = requests.post(url, headers=headers, files=files)

#print("Uploaded: "+str(response.text))