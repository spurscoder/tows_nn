import sys

import jsonlines
import json


def generator(item1):
    with open("./dataset/ws3.jl") as f2:
        id = 0
        print("一轮遍历")
        while 1:
            id += 1
            data = {}
            item = jsonlines.Reader(f2).iter()
            item2 = next(item)
            data["id"] = id
            data["intro1"] = item1["intro"]
            data["intro2"] = item2["intro"]
            data["tag"] = gentag(item1["tag"], item2["tag"])  # 判断tag
            yield data


def gendataset():
    dataset = []
    data = {}
    id = 0
    with open("./dataset/ws2.jl") as f1:
        item = jsonlines.Reader(f1).iter()
        gen = generator(next(item))
        yield next(gen)


def gentag(tagset_1, tagset_2):
    tag = 0
    for item in tagset_1:
        if item in tagset_2:
            tag = 1
    return tag


if __name__ == "__main__":
    num = 30000
    datalist = []

    datagen = gendataset()
    id = 0
    with jsonlines.open("./dataset/ws6.jl", "a") as f:
        with open("./dataset/ws2.jl") as f1:
            with open("./dataset/ws3.jl") as f2:
                for item1 in jsonlines.Reader(f1):
                    intro = item1["intro"]
                    f2.seek(0)
                    for item2 in jsonlines.Reader(f2):
                        data = {}
                        id += 1
                        data["id"] = id
                        data["intro1"] = intro
                        data["intro2"] = item2["intro"]
                        data["tag"] = gentag(item1["tag"], item2["tag"])  # 判断tag
                        jsonlines.Writer.write(f, data)
                        print(id)
                    if id > 1000000:
                        sys.exit()
    # try:
    #     with jsonlines.open('./dataset/ws4.jl', 'w') as f:
    #         for i in range(21896):
    #             for j in gendataset():
    #                 # datagenerator = next(datagen)
    #             # id = data.get("id")
    #                 jsonlines.Writer.write(f, j)
    # except StopIteration:
    #     print(11111111)
