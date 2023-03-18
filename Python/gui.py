from tkinter import Tk, ttk, Label, Button, Text, EXTENDED

stationfares = [
    {"station": "union", "price": 1.5},
    {"station": "redline", "price": 1.5},
    {"station": "iit", "price": 2.0},
    {"station": "lake", "price": 1.0},
]

selected_index =0

window = Tk()
window.title("Station Fares")
window.geometry("500x500")
window.resizable(0, 0)
title = "Station Fares"
lbl_title = Label(window, text=title)
lbl_title.pack(padx=5, pady=15)


treeStationFares = ttk.Treeview(window)
treeStationFares["columns"] = ("Station", "Price")
treeStationFares.column("#0", width=0)
treeStationFares.column("Station", width=100)
treeStationFares.column("Price", width=100)

treeStationFares.heading("#0", text="Turn")
treeStationFares.heading("Station", text="Station")
treeStationFares.heading("Price", text="Price")
treeStationFares.place(x=100, y=100, width=400, height=200)

def stationfares_selection():
    global selected_index
    for item in treeStationFares.selection():
        selected_index = int(treeStationFares.item(item, "text"))
    stationfare = stationfares[selected_index]
    station = stationfare["station"]
    price = str(stationfare["price"])
    text_Station.delete("1.0", END)
    text_Station.insert("END", station)
    text_price.delete("1.0", END)
    text_price.insert("END", price)

def setTreeview():
    treeStationFares.delete(*treeStationFares.get_children())
    for idx, stationfare in enumerate(stationfares):
        station= stationfare['station']
        price = str(stationfare['price'])
        treeStationFares.insert("", "end", text=str(idx), values=(station, price))
def insert_command():

    station = text_Station.get("1.0", END)
    price = int(text_price.get("1.0", END))
    stationfares.append({"station": station, "price": price})
    setTreeview()

def update_command():
    global selected_index
    station = text_Station.get("1.0", END)
    price = int(text_price.get("1.0", END))
    selectedItem = stationfares[selected_index]
    selectedItem["station"] = station
    selectedItem["price"] = price
    setTreeview()

def delete_command():
    global selected_index
    stationfares.pop(selected_index)
    setTreeview()

treeStationFares.bind("<<TreeviewSelect>>", stationfares_selected)


btn_Insert = Button(window, text="Insert", command=insert_content)
btn_Insert.place(x=100, y=300, width=100, height=50)

btn_Update = Button(window, text="Update", command=update_content)
btn_Update.place(x=100, y=300, width=100, height=50)

btn_Delete = Button(window, text="Delete", command=delete_content)
btn_Delete.place(x=100, y=300, width=100, height=50)


labelStation = Label(window, text="Station")
labelStation.place(x=100, y=400, width=100, height=50)
labelFare = Label(window, text="Fare")
labelFare.place(x=100, y=450, width=100, height=50)
text_Station = Text(window, height=5, width=100)
text_Station.place(x=100, y=500, width=100)
text_price = Text(window, height=5, width=100)
text_price.place(x=100, y=550, width=100)

setTreeItems()


window.mainloop()
