import ask_ai

sentence = 'Hi, I am GroceryLister. Lets talk!'
print(sentence)

name = input('What would you like to be called? \n')

print('Nice to meet you', name)
age = input('How old are you? \n')
print('Wow,',age, 'is a great age' )
print('Lets talk some more')

data={"CITY (IN THE BAY AREA)":"San Jose","When do you want it delivered (days)(0 if you don't want delivery)":0,"BUDGET (max $ per item)":27,"How important is healthy food?(out of 10)":9,"Price for shipping ($) (0 if you don't want delivery)":1,"Delivery Yes or No":"No"}

mood = ask_ai.get_prediction(data)
print(mood)

if mood == 'FoodMaxx':
    print('The best grocery store would be FoodMaxx.')
elif mood == 'Fresco Supermarket':
    print('The best grocery store would be Fresco Supermarket.')
elif mood == 'Grocery Outlet':
    print('The best grocery store would be Grocery Outlet.')
elif mood == 'H Mart':
    print('The best grocery store would be H Mart.')
elif mood == 'Lunardis Market':
    print('The best grocery store would be Lunardis Market.')
elif mood == 'Nob Hill foods':
    print('The best grocery store would be Nob Hill foods.')
elif mood == 'Real Produce':
    print('The best grocery store would be Real Produce.')
elif mood == 'Safeway':
    print('The best grocery store would be safeway.')
elif mood == 'Smart & Final':
    print('The best grocery store would be Smart & Final.')
elif mood == 'Sprouts Farmers Market':
    print('The best grocery store would be Sprouts Farmers Market.')
elif mood == 'Target':
    print('The best grocery store would be Target.')
elif mood == 'trader joes':
    print('The best grocery store would be trader joes.')
elif mood == 'Whole Foods':
    print('The best grocery store would be Whole Foods.')
elif mood == 'Zanottos Family Market':
    print('The best grocery store would be Zanottos Family Market.')
elif mood == 'Sprouts Farmers Market':
    print('The best grocery store would be Sprouts Farmers Market.')
elif mood == 'Whole Foods':
    print('The best grocery store would be Whole Foods.')

 
else:
    print('Sorry, I dont understand, can you tell me more?')