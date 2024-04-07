import os
import glob

'''
数据预处理：从pcap文件中提取特征值
'''

##########################

# IP 过滤条件
ip_filter = {}

ip_filter['TCP_Mobile'] = "'tcp && (ip.src==192.168.1.45)'"
ip_filter['TCP_Outlet'] = "'tcp && (ip.scr==192.168.1.222) || (ip.src==192.168.1.67)'"
ip_filter['TCP_Assistant'] = "'tcp && (ip.src==192.168.1.111) || (ip.src==192.168.1.30) || (ip.src==192.168.1.42)" \
                             " || (ip.src==192.168.1.59) || (ip.src==192.168.1.70)'"
ip_filter['TCP_Camera'] = "'tcp && (ip.src==192.168.1.128) || (ip.src==192.168.1.145) || (ip.src==192.168.1.78)'"
ip_filter['TCP_Miscellaneous'] = "'tcp && (ip.src==192.168.1.46) || (ip.src==192.168.1.84) || (ip.src==192.168.1.91)'"

##########################

# 声明数据存储文件
labelFeature = open("label_feature_IOT.csv", 'a')

# 设置数据标签
labelFeature.writelines("Label,IPLength,IPHeaderLength,IPFlags,TTL,Protocol,IPID,IPchecksum,"
                        "SourcePort,DestPort,SequenceNumber,AckNumber,WindowSize,TCPHeaderLength,"
                        "TCPflags,TCPLength,TCPChecksum,TCPStream,TCPUrgentPointer\n")

# 处理文件
for original in glob.glob('original_pcap/*.pcap'):
    for k in ip_filter.keys():
        os.system("tshark -r" + original + " -w- -Y "+ ip_filter[k] + ">> filter_pcap/" + k + ".pcap")

# 提取文件信息存储
for filteredFile in glob.glob('filtered_pcap/*.pcap'):
    print(filteredFile)
    filename = filteredFile.split('/')[-1]
    label = filename.replace(".pcap", '')
    tsharkCommand = "D:\Software\Wireshark\\tshark.exe -r " + filteredFile + " -T fields \
                     -e ip.len -e ip.hdr_len -e ip.flags -e ip.ttl \
                     -e ip.proto -e ip.id -e ip.checksum -e tcp.srcport -e tcp.dstport \
                     -e tcp.seq -e tcp.ack -e tcp.window_size_value -e tcp.hdr_len \
                     -e tcp.flags -e tcp.len -e tcp.checksum -e tcp.stream \
                     -e tcp.urgent_pointer"

    allFeatures = str( os.popen(tsharkCommand).read() )
    allFeatures = allFeatures.replace('\t',',')
    allFeaturesList = allFeatures.splitlines()
    # print(allFeatures)
    for features in allFeaturesList:
        labelFeature.writelines(label + "," + features + "\n")

# for filteredFile in glob.glob('filtered_pcap/*.pcap'):
#     #print(filteredFile)
#     filename = filteredFile.split('/')[-1]
#     label = filename.replace('.pcap', '')
#     tsharkCommand = "D:\Software\Wireshark\\tshark.exe -r " + filteredFile + " -T fields \
#                     -e ip.len -e ip.hdr_len -e ip.ttl \
#                     -e ip.proto -e tcp.srcport -e tcp.dstport -e tcp.seq \
#                     -e tcp.ack -e tcp.window_size_value -e tcp.hdr_len -e tcp.len \
#                     -e tcp.stream -e tcp.urgent_pointer \
#                     -e ip.flags -e ip.id -e ip.checksum -e tcp.flags -e tcp.checksum"
#
#     allFeatures = str(  os.popen(tsharkCommand).read()  )
#     allFeatures = allFeatures.replace('\t',',')
#     allFeaturesList = allFeatures.splitlines()
#     for features in allFeaturesList:
#         labelFeature.writelines(label + "," + features + "\n")

###########################
print("DONE")