import xlwt
import codecs

input_txt = r'E:\ProgramData\Anaconda3\envs\minglangtf\eyolo3-keras-master\results\rec.txt'
output_excel = r'E:\ProgramData\Anaconda3\envs\minglangtf\eyolo3-keras-master\results\rec.xls'
sheetName = 'Sheet1'
start_row = 0
start_col = 0

wb = xlwt.Workbook(encoding = 'utf-8')
ws = wb.add_sheet(sheetName)

f = open(input_txt, encoding = 'utf-8')

col_excel = start_col
for line in f:
    line = line.strip('\n')
    line = line.split(',')
    
    print(line)
    
    row_excel = start_row
    len_line = len(line)
    for j in range(len(line)):
        #print (line[j])
        
        ws.write(row_excel,col_excel,line[j])
        row_excel += 1
        wb.save(output_excel)

    col_excel += 1

f.close
