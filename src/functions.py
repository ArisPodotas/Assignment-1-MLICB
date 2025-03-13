import argparse as arg

def liner() -> arg.Namespace:
    """Prses the command line input"""
    prs = arg.ArgumentParser()
    cmd = prs.parse_args()
    return cmd

def main():
    pass

if __name__ == "__main__":
    main()

