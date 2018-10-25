def dat2csv(dem):
    data = dem.read()
    data = data.split('\n')
    demcsv = open('dem.csv', 'w')
    demcsv.write('x,y,z\n')
    for lines in data:
        line = lines.split('\t')
        if len(line) == 3:
            x = line[0].strip()
            y = line[1].strip()
            z = line[2].strip()
            demcsv.write(x + ',' + y + ',' + z + '\n')
    demcsv.close()


dem = open('lonlathei.dat', 'r')
dat2csv(dem)
dem.close()
