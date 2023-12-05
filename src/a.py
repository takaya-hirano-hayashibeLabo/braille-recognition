import argparse


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--list",nargs="*",type=str)
    args=parser.parse_args()
    
    print(args.list)
    
if __name__=="__main__":
    main()