import pdb

# Some loop
for i in range(5):
    print("Starting iteration", i)
    
    # Your code here
    
    # Set a breakpoint to pause execution and enter pdb
    pdb.set_trace()
    
    # After reaching the breakpoint, you can interact with pdb
    # Type 'c' and press Enter to continue to the next iteration
    # Type 'q' and press Enter to exit the debugger
    
    print("Ending iteration", i)

    
