import importlib


def main():
    print("Available use cases:")
    print("1 - use_case_1")
    print("2 - use_case_2")
    print("app1 - flask_apps.app1")
    print("app2 - flask_apps.app2")
    print("app3 - flask_apps.app3")
    choice = input("Enter use case to run: ").strip()

    # Mapping choices to module paths
    use_case_map = {
        '1': 'use_cases.use_case_1',
        '2': 'use_cases.use_case_2',
        'app1': 'use_cases.flask_apps.app1.server',
        'app2': 'use_cases.flask_apps.app2.server',
        'app3': 'use_cases.flask_apps.app3.server',
    }

    module_name = use_case_map.get(choice)
    if not module_name:
        print("Invalid choice!")
        return

    module = importlib.import_module(module_name)
    # Expect each module to have a run() function
    module.run()


if __name__ == "__main__":
    main()
