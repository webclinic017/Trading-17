import sys
import trainer

args = sys.argv[1:]
index = 1

'''
Config values for the correct run
'''
Strategy_file = args[0]
Data_file = None
Mode = None
AI = False

'''
Set config for run
'''
while index != len(args):
    if args[index] in ['--test', '--train']:
        if Mode == None:
            Mode = args[index]
            index += 1
        else:
            raise RuntimeError('Multiple modes set')

    elif args[index] in ['--AI']:
        AI = True
        index += 1

    elif args[index] == '--on-file':
        try:
            Data_file = args[index+1]
            index += 2
        except IndexError:
            raise RuntimeError('Need to specify the data file to read after --on-file')

    else:
        raise RuntimeError('{} - Is not a valid option'.format(args[index]))


'''
Execute run with config
'''
# Train AI stratergy on a data file
if (Strategy_file != None and Data_file != None and Mode == '--train' and AI == True):
    trainer.main_AI(Strategy_file, Data_file)


else:
    raise RuntimeError('Something went wrong')