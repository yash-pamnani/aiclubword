import requests
import json
def get_prediction(data={"CITY (IN THE BAY AREA)":"San Jose","When do you want it delivered (days)(0 if you don't want delivery)":0,"BUDGET (max $ per item)":17,"How important is healthy food?(out of 10)":4,"Price for shipping ($) (0 if you don't want delivery)":0,"Delivery Yes or No":"Yes"}):
  url = 'https://askai.aiclub.world/8f3e9c4f-d530-4558-a890-1febb9594428'
  r = requests.post(url, data=json.dumps(data))
  response = getattr(r,'_content').decode("utf-8")
  
 
  prediction = json.loads(json.loads(response)['body'])['predicted_label']
  print(prediction)
  return prediction