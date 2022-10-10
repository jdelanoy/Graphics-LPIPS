import csv

res_folder="checkpoints/YanaParam_baseline_k0/test_best_light_y_%i/GraphicsLPIPS_global.csv"

list_l1 = []
list_l2 = []
list_pearson = []
list_spearman = []

for i in range (12):
    csv_file = res_folder%i
    with open(csv_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row[0] == "l1": list_l1.append(float(row[1]))
            elif row[0] == "l2": list_l2.append(float(row[1]))
            elif "pearson" in row[0] : list_pearson.append(float(row[1]))
            elif "spearman" in row[0] : list_spearman.append(float(row[1]))

print("l1",list_l1)
print("l2",list_l2)
print("pearson",list_pearson)
print("spearman",list_spearman)

from matplotlib import pyplot as plt
plt.plot(range(-30,90,10),list_pearson,label="pearson")
plt.plot(range(-30,90,10),list_spearman,label="spearman")
plt.legend()
plt.show()