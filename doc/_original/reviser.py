import os
import glob2


# =============================================================================
# reading absolute directory for this python file
# =============================================================================
python_path=os.path.abspath(__file__)
doc_dir="/".join(python_path.split("/")[:-2])+"/_locale/ja/LC_MESSAGES"
po_list=list(glob2.iglob(doc_dir+"/**/*.po",recursive=True))




# =============================================================================
# write script
# =============================================================================

for po in po_list:

    base_po="/".join(po.split("/")[len(doc_dir.split("/")):])

    #read files
    with open(po, 'r') as f:
        string_list = f.readlines()

    #set intial parameters
    i=0
    msg_str=False
    res=[]
    count_a, count_b = 0, 0



    while i < len(string_list):

        msg=string_list[i]

        if "msgstr" in msg:
            msg_str=True

        if msg=="\n":
            msg_str=False

        if msg_str==True:

            count_a += msg.count("。")
            count_b += msg.count("、")

            msg=msg.replace("。",". ").replace("、",", ")
            msg=msg.replace(".  ",". ").replace(",  ",", ")

        i += 1

        res.append(msg)


    print("{} - overwrite  '、':{}  '。':{} ".format(base_po,count_a,count_b))

    #saving file
    with open(po,'w') as f:
        f.write("".join(res))
        f.close()









# print(string_list)
# lc_dir=[]
# for po in po_list:
#     lc_dir.append("/".join(po.split("/")[len(doc_dir.split("/")):]))
