contacts = {}
with open('contacts.txt', 'r') as file:
    for li in file:
        parts = li.strip().split(',')
        if len(parts) >=2:
            name= parts[0]
            info= parts[1:]
            contacts[name]=info

print('Contactsdictionary')
for name in contacts:
    print(name, contacts[name])