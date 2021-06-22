# For one-hot lookup table

norm_spks = [
    'BZNSYP', 
]

aishell3_spks = [
    'SSB0005', 'SSB0009', 'SSB0011', 'SSB0012', 'SSB0016', 'SSB0018', 'SSB0033', 'SSB0038', 'SSB0043', 'SSB0057', 
    'SSB0073', 'SSB0080', 'SSB0112', 'SSB0122', 'SSB0133', 'SSB0139', 'SSB0145', 'SSB0149', 'SSB0193', 'SSB0197',
    'SSB0200', 'SSB0241', 'SSB0246', 'SSB0261', 'SSB0267', 'SSB0273', 'SSB0287', 'SSB0288', 'SSB0299', 'SSB0307', 
    'SSB0309', 'SSB0315', 'SSB0316', 'SSB0323', 'SSB0338', 'SSB0339', 'SSB0341', 'SSB0342', 'SSB0354', 'SSB0366', 
    'SSB0375', 'SSB0379', 'SSB0380', 'SSB0382', 'SSB0385', 'SSB0393', 'SSB0394', 'SSB0395', 'SSB0407', 'SSB0415', 
    'SSB0426', 'SSB0427', 'SSB0434', 'SSB0435', 'SSB0470', 'SSB0482', 'SSB0502', 'SSB0534', 'SSB0535', 'SSB0539', 
    'SSB0544', 'SSB0565', 'SSB0570', 'SSB0578', 'SSB0588', 'SSB0590', 'SSB0594', 'SSB0599', 'SSB0601', 'SSB0603', 
    'SSB0606', 'SSB0607', 'SSB0609', 'SSB0614', 'SSB0623', 'SSB0629', 'SSB0631', 'SSB0632', 'SSB0666', 'SSB0668', 
    'SSB0671', 'SSB0686', 'SSB0693', 'SSB0700', 'SSB0702', 'SSB0710', 'SSB0711', 'SSB0716', 'SSB0717', 'SSB0720', 
    'SSB0723', 'SSB0736', 'SSB0737', 'SSB0746', 'SSB0748', 'SSB0749', 'SSB0751', 'SSB0758', 'SSB0760', 'SSB0762', 
    'SSB0778', 'SSB0780', 'SSB0784', 'SSB0786', 'SSB0794', 'SSB0809', 'SSB0817', 'SSB0822', 'SSB0851', 'SSB0863', 
    'SSB0871', 'SSB0887', 'SSB0913', 'SSB0915', 'SSB0919', 'SSB0935', 'SSB0966', 'SSB0987', 'SSB0993', 'SSB0997', 
    'SSB1000', 'SSB1001', 'SSB1002', 'SSB1008', 'SSB1020', 'SSB1024', 'SSB1050', 'SSB1055', 'SSB1056', 'SSB1064', 
    'SSB1072', 'SSB1091', 'SSB1096', 'SSB1100', 'SSB1108', 'SSB1110', 'SSB1115', 'SSB1125', 'SSB1126', 'SSB1131', 
    'SSB1135', 'SSB1136', 'SSB1138', 'SSB1161', 'SSB1176', 'SSB1187', 'SSB1197', 'SSB1203', 'SSB1204', 'SSB1215', 
    'SSB1216', 'SSB1218', 'SSB1219', 'SSB1221', 'SSB1239', 'SSB1253', 'SSB1274', 'SSB1302', 'SSB1320', 'SSB1322', 
    'SSB1328', 'SSB1340', 'SSB1341', 'SSB1365', 'SSB1366', 'SSB1377', 'SSB1382', 'SSB1383', 'SSB1385', 'SSB1392', 
    'SSB1393', 'SSB1399', 'SSB1402', 'SSB1408', 'SSB1431', 'SSB1437', 'SSB1448', 'SSB1452', 'SSB1457', 'SSB1555', 
    'SSB1563', 'SSB1567', 'SSB1575', 'SSB1585', 'SSB1593', 'SSB1607', 'SSB1624', 'SSB1625', 'SSB1630', 'SSB1650', 
    'SSB1670', 'SSB1684', 'SSB1686', 'SSB1699', 'SSB1711', 'SSB1728', 'SSB1739', 'SSB1745', 'SSB1759', 'SSB1781', 
    'SSB1782', 'SSB1806', 'SSB1809', 'SSB1810', 'SSB1828', 'SSB1831', 'SSB1832', 'SSB1837', 'SSB1846', 'SSB1863', 
    'SSB1872', 'SSB1878', 'SSB1891', 'SSB1902', 'SSB1918', 'SSB1935', 'SSB1939', 'SSB1956', 
]

speakers = aishell3_spks
speaker_to_id = {s: i for i, s in enumerate(speakers)}