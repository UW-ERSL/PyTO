from solidworks_interface import SolidWorksInterface

input("Is SolidWorks open with a TwoHolesTang?\nPress Enter to continue...")
solidWorks = SolidWorksInterface()
dimName1 = "D7@Sketch1"
dim1Default = 0.019

dimName2 = "D8@Sketch1"
dim2Default = 0.034


success = solidWorks.setDimensions({dimName1,dimName2}, {dim1Default,dim2Default})


success = solidWorks.setDimension(dimName1, 0.018)
success = solidWorks.setDimension(dimName2, 0.034)
# dim1Range = [0.015 + i*(0.025 - 0.015)/20 for i in range(21)]
# dim2Range = [0.030 + i*(0.040 - 0.030)/20 for i in range(21)]
# print(f"Dimension ranges: {dim1Range}, {dim2Range}")
# input("Press Enter to start setting dimensions...")
# results = []

# for i in range(len(dim1Range)):
#     d1 = dim1Range[i]
#     d2 = dim2Range[i]
#     print(d1,d2)
#     success = solidWorks.setDimensions({dimName1,dimName2}, {d1,d2})

#     if not success:
#         results.append((d1, d2, False))
#         solidWorks.setDimensions({dimName1,dimName2}, {dim1Default,dim2Default})
#         input("Failure: Press Enter to continue to next dimension pair...")
#         continue
  
#     input("Success: Press Enter to continue to next dimension pair...")
# print(results)