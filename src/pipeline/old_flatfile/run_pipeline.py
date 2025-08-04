from pipeline import data_loader as loader
from pipeline import data_parser as parser
from pipeline import data_aggregator as aggregator
from pipeline import data_preprocessor as preprocessor


def main():
    exit = False

    while (not exit):

        print("\nSelect an option:\n")
        print("1. Loader")
        print("2. Parser")
        print("3. Aggregator")
        print("4. Preprocessor")
        print("5. All")
        print("6. Exit")

        choice = input("> ")

        if choice == '1':
            loader.run()
        elif choice == '2':
            parser.run()
        elif choice == '3':
            aggregator.run()
        elif choice == '4':
            preprocessor.run()
        elif choice == '5':
            loader.run()
            parser.run()
            aggregator.run()
            preprocessor.run()
        elif choice == '6':
            exit = True
        else:
            print("\nInvalid output...\n")

if __name__ == "__main__":
    main()
