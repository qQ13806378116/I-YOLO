from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString
import os
import json
 
def make_xml(coco_json, save_xml_path):
    name = []
    img_bbox = {}
    bbox_category = {}
    for line in coco_json:
        # 获取图片名字
        img_name = str(line['image_id']).strip() + '.jpg'
        # 获取目标框信息
        bbox = line['bbox']
        category = line['category_id']
 
        bbox_category[str(bbox)] = category
        if img_name not in name:
            name.append(img_name)
            img_bbox[img_name] = [bbox]
        else:
            img_bbox[img_name].append(bbox)
 
    # 第一层循环遍历所有的照片
    for img_name in img_bbox.keys():
        print(img_name)
        # 获取图片名字
        pic_name = img_name
 
        node_root = Element('annotation')
        node_filename = SubElement(node_root, 'filename')
        # 图片名字
        node_filename.text = pic_name
 
        # 第二层循环遍历有多少个框
        for bbox in img_bbox[img_name]:
            category_name = bbox_category[str(bbox)]
 
            node_object = SubElement(node_root, 'object')
            node_name = SubElement(node_object, 'name')
            # 类别名字
            node_name.text = str(int(category_name)-1)
 
            node_bndbox = SubElement(node_object, 'bndbox')
            node_xmin = SubElement(node_bndbox, 'xmin')
            node_xmin.text = str(int(bbox[0]))
            node_ymin = SubElement(node_bndbox, 'ymin')
            node_ymin.text = str(int(bbox[1]))
            node_xmax = SubElement(node_bndbox, 'xmax')
            node_xmax.text = str(int(bbox[0])+int(bbox[2]))
            node_ymax = SubElement(node_bndbox, 'ymax')
            node_ymax.text = str(int(bbox[1])+int(bbox[3]))
 
        xml = tostring(node_root, pretty_print=True)
        dom = parseString(xml)
        # print xml 打印查看结果
        xml_name = pic_name.replace(".jpg", "")
        xml_name = os.path.join(save_xml_path, xml_name + '.xml')
        with open(xml_name, 'wb') as f:
            f.write(dom.toprettyxml(indent='\t', encoding='utf-8'))
 
 
# json文件路径
path = "E:\learning\dataset\FLIR_ADAS_1_3 (2)\FLIR_ADAS_1_3/train/1.json"
file = open(path, "r", encoding='utf-8')
fileJson = json.load(file)
field = fileJson["annotations"]
 
save_xml_path = 'E:\learning\dataset\FLIR_ADAS_1_3 (2)\FLIR_ADAS_1_3/train/XML'
make_xml(field, save_xml_path)