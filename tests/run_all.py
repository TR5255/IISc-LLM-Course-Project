import test_router
import test_dataset

if __name__ == "__main__":
    print("Running verification tests...")
    
    try:
        # Router unit tests
        test_router.test_config_namespace()
        print("✓ test_config_namespace passed")
        
        test_router.test_scorers_output_range()
        print("✓ test_scorers_output_range passed")
        
        test_router.test_routing_policies()
        print("✓ test_routing_policies passed")
        
        test_router.test_router_flow()
        print("✓ test_router_flow passed")
        
        test_router.test_f1_calculation()
        print("✓ test_f1_calculation passed")
        
        test_router.test_ndcg_calculation()
        print("✓ test_ndcg_calculation passed")
        
        # Dataset validation unit tests
        test_dataset.test_benchmark_data_validation()
        print("✓ test_benchmark_data_validation passed")
        
        test_dataset.test_random_baseline_execution()
        print("✓ test_random_baseline_execution passed")
        
        test_dataset.test_all_scorers_benchmark_run()
        print("✓ test_all_scorers_benchmark_run passed")
        
        print("\nAll tests passed successfully!")
    except AssertionError as e:
        print(f"Assertion failed: {e}")
        exit(1)
    except Exception as e:
        print(f"Error during tests: {e}")
        exit(1)
