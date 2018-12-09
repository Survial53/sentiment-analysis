import datetime

now = datetime.datetime.now()

out_name = "Bag_of_Words_model_" + now.strftime("%Y-%m-%d %H:%M") + ".csv"

print(out_name)