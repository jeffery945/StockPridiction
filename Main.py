from tkinter import *
import numpy as np
import pandas as pd
import requests
import json
import datetime
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
current_dateTime = datetime.datetime.now()
#function

# 獲取收盤價與成交股數data
def CollectPriceData():
    stock_number = str(Stock_Number_entry.get())
    final_price = []
    stock_amount = []
    start_dateYear = 2022   #因為data是從2022年開始爬
    start_dateMonth = 1
    while not (start_dateYear >= current_dateTime.year and start_dateMonth > current_dateTime.month):

        date = str(start_dateYear)+str(start_dateMonth).zfill(2)+"01"#先轉為20220101的形式
        html = requests.get(
            'https://www.twse.com.tw/exchangeReport/STOCK_DAY?response=json&date=%s&stockNo=%s' % (date, stock_number))
        content = json.loads(html.text)
        stockData = content['data'] #json檔爬下來後做data跟fields的分類，使得有columns也有data
        attribute = content["fields"]
        dataFrame_price = pd.DataFrame(data=stockData, columns=attribute)#建成dataframe比較清楚
        dataFrame_price.head()
        for i in range(0, (len(dataFrame_price["收盤價"].values.tolist())), 5): #因為如果從202201開始爬每一天的收盤價會很龐大而且雜亂
            final_price.append(                                                 #所以我決定以每周平均值計算
                np.mean(list(map(float, dataFrame_price["收盤價"].values.tolist()[i:i+5]))))
            temp = dataFrame_price["成交股數"].values.tolist()[i:i+5]           #所以成交股數我也用每周mean下去算
            temp = [element.replace(",", "") for element in temp]
            stock_amount.append(np.mean(list(map(int, temp))))

        start_dateMonth += 1
        if (start_dateMonth == 12):
            start_dateMonth = 1
            start_dateYear += 1
    return final_price, stock_amount 


#獲取資金流向data
def CollectCapitalData():
    df_array = np.zeros(30) #因為證交所網站有固定三十項指數，所以先建好
    dateCount = 1
    start_dateYear = 2023
    start_dateMonth = 1 if current_dateTime.month == 1 else current_dateTime.month - 1
    while ( start_dateMonth <= current_dateTime.month ):
        if(dateCount >= current_dateTime.day and start_dateMonth == current_dateTime.month):#如果大於現在的date了就跳出回圈
            break
        date = str(start_dateYear)+str(start_dateMonth).zfill(2)+str(dateCount).zfill(2)
        html = requests.get(
            'https://www.twse.com.tw/rwd/zh/afterTrading/BFIAMU?response=json&date=%s' % (date))
        content = json.loads(html.text)
        if(content["stat"]!="OK"):#因為有些天數是沒開市的要跳過
            dateCount+=1
            continue
        if(dateCount >= 29):
            start_dateMonth+=1
            dateCount = 1
        capitalData = content["data"]
        attribute_f = content["fields"]
        dataFrame = pd.DataFrame(data=capitalData, columns=attribute_f)
        dataFrame.head()
        array_df_temp = np.array(dataFrame['漲跌指數'].values.tolist())
        array_df_temp = array_df_temp.astype(float)
        df_array += array_df_temp   #把指數漲跌都相加
        dateCount += 1
    
    Capital_dic = zip(dataFrame['分類指數名稱'].values.tolist(), df_array)#zip成一個dictionary
    print(Capital_dic)
    sorted_items = sorted(list(Capital_dic), key=lambda x:x[1], reverse=True)#從大到小排好，才可以找出前三名
    name = [item[0] for item in sorted_items]           #分成名字與指數值在UI上顯示
    values = [item[1] for item in sorted_items]
    value = [round(value, 2) for value in values]
    capital_flows_first_name.config(text=f"{name[0]}")  #以下就是顯示在UI上
    capital_flows_second_name.config(text=f"{name[1]}")
    capital_flows_third_name.config(text=f"{name[2]}")
    capital_flows_first_value.config(text=f"{value[0]}")
    capital_flows_second_value.config(text=f"{value[1]}")
    capital_flows_third_value.config(text=f"{value[2]}")


#預測股價
def Predict_Price():
    result_price = CollectPriceData()   #取得final_price與stock_amount
    result_price_amount = []            #因為tuple不能reshape所以轉成list
    result_price_amount = list(result_price[1])
    result_price_amount = np.reshape(
        result_price_amount, (len(result_price_amount), 1))
    x_train, x_test, y_train, y_test = train_test_split(                        #利用train_test_split分成0.75train與0.25test
        result_price_amount,result_price[0], test_size=0.25, random_state=0)

    model = SVR(C=1e3, kernel='rbf', gamma=0.0001)                              #套用SVR來做機器學習回歸分析
    model.fit(x_train, y_train)
    prediction = model.predict([[int(Stock_Amount_entry.get())]])               #取得輸入的amount下去做預測
    print("Tomorrow's predicted stock price:", prediction)
    accuracy = model.score(x_train, y_train)
    print(f"And the accuracy is :{accuracy}")                                   #正確率
    stock_price_output.config(text=f"{round(prediction[0], 2)}")
    CollectCapitalData()                                                        #因為按下按鈕就要連Capital都顯示，所以要call CollectCapitalData



# window

stock_window = Tk()
stock_window.title("運用我來發大財吧")
stock_window.minsize(width=400, height=400)
stock_window.config(padx=20, pady=20, background="#00BFFF")

#label
stock_price_title = Label(width=18, height=1,text="透過成交量預測收盤價",fg="#7FFFD4",font=("Arial", 17, "bold"), background="#00BFFF")
stock_price_title.place(x = 58, y= 0)

stock_price_number = Label(anchor="nw",width=12, height=1,text="Stock Number",fg="#0000CD",font=("Arial", 13, "normal"), background="#00BFFF")
stock_price_number.place(x=60 ,y= 50)

stock_price_amount = Label(anchor="nw",width=12, height=1,text="Stock Amount",fg="#0000CD",font=("Arial", 13, "normal"), background="#00BFFF")
stock_price_amount.place(x=60 ,y= 90)

stock_price = Label(anchor="nw",width=12, height=1,text="Predict Price :",fg="#0000CD",font=("Arial", 13, "normal"), background="#00BFFF")
stock_price.place(x=60 ,y= 170)

stock_price_output = Label(anchor="nw",width=12, height=1,fg="#0000CD",font=("Arial", 13, "normal"), background="#00BFFF")
stock_price_output.place(x=240 ,y= 170)

capital_flows_title = Label(width=18, height=1,text="近一個月指數漲幅前三",fg="#7FFFD4",font=("Arial", 17, "bold"), background="#00BFFF")
capital_flows_title.place(x = 58, y= 220)

capital_flows_first_name = Label(width=14, height=1,fg="#0000CD",font=("Arial", 11, "normal"), background="#00BFFF")
capital_flows_first_name.place(x = 0, y= 285)

capital_flows_second_name = Label(width=14, height=1,fg="#0000CD",font=("Arial", 11, "normal"), background="#00BFFF")
capital_flows_second_name.place(x = 125, y= 285)

capital_flows_third_name = Label(width=14, height=1,fg="#0000CD",font=("Arial", 11, "normal"), background="#00BFFF")
capital_flows_third_name.place(x = 250, y= 285)

capital_flows_first_value = Label(width=14, height=1,fg="#0000CD",font=("Arial", 11, "normal"), background="#00BFFF")
capital_flows_first_value.place(x = 0, y= 330)

capital_flows_second_value = Label(width=14, height=1,fg="#0000CD",font=("Arial", 11, "normal"), background="#00BFFF")
capital_flows_second_value.place(x = 125, y= 330)

capital_flows_third_value = Label(width=14, height=1,fg="#0000CD",font=("Arial", 11, "normal"), background="#00BFFF")
capital_flows_third_value.place(x = 250, y= 330)

#entry

Stock_Number_entry = Entry(width=9, background="#87CEEB")
Stock_Number_entry.place(x=240, y=50)

Stock_Amount_entry = Entry(width=9, background="#87CEEB")
Stock_Amount_entry.place(x=240, y=90)

#button
Pridict_Button = Button(width=33, text="Pridict Price",font=("Arial", 9, "bold"),background="#FF7F50", activeforeground="#DC143C", command=Predict_Price)
Pridict_Button.place(x=65, y=130)


stock_window.mainloop()         #讓畫面停止

