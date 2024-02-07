import argparse

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--is_view",action="store_true")
    args=parser.parse_args()

    print(args.__dict__)

    if args.is_view:
        print("Hello world")

if __name__=="__main__":
    main()