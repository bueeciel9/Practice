class Comments:
    title = "Comments"
    product_description = "This is a product description"
    insert_fare = "This is a fare"
    insufficient_fare = "Insufficient fare"
    select_station = "Select station"
    select_error = "Select error"
    finish_sale = "Finish sale"
    terminate_sale = "Terminate sale"

class Products:
    productName = []
    productValues = []

class SubwayTicket(Products):
    _data = stationfares
    _name = "Subway Ticket"
    _status = True

    def __init__(self):
        print(Comments.title %self._name)
        self._fare = 0
        self._station = 0
    def set_products(self):
        Products.productNames = []
        Products.productValues = []
        for stationsfare in self._data:
            Products.productNames.append("station")
            Products.productValues.append("fare")

    def run(self):
        self.set_products()
        while self._status:
            try:
                inputMoney = int(input(Comments.insert_fare))
            except ValueError:
                print(Comments.select_error)
            else:
                self.selectStation(inputMoney)
    def selectStation(self, money):
        for idex, name in enumerate(Products.productNames):
            fare = Products.productValues[idex]
            print("%s: %s" %(name, fare))
        inputStation = input(Comments.select_station)
        self.payment(money, inputStation)
    
    def payment(self, money, idx):
        name = Products.productNames[idx]
        fare = Products.productValues[idx]
        if money >= fare:
            balance = money - fare
            print(Comments.product_description %name)
        else:
            print(Comments.insufficient_fare)
            balance = money

tm = SubwayTicket()

try:
    tm.run()
except KeyboardInterrupt:
    tm._statue = False
    print(Comments.terminate_sale)
