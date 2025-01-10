Creational Pattern: for object creation
-
<details>
  <summary>Singleton</summary>
  
  - only one object accessible globally
  - Singleton's constructor/destructor should always be private to prevent direct construction/desctruction calls with the `new`/`delete` operator
  ```cpp
  class Singleton {
  private:
      static Singleton* instancePtr; // Static pointer to the Singleton instance
      static mutex mtx; // Mutex to ensure thread safety
      Singleton() {}
  
  public:
      // Deleting the copy or assign constructor: https://cplusplus.com/doc/tutorial/classes2/
      Singleton(const Singleton& obj) = delete;
      Singleton& operator=(const Singleton&) = delete;
  
      static Singleton* getInstance() { // Static method to get the Singleton instance
          if (instancePtr == nullptr) {
              //lock_guard<mutex> lock(mtx);
              if (instancePtr == nullptr) instancePtr = new Singleton();
          }
          return instancePtr;
      }
  };
  
  Singleton* Singleton::instancePtr = nullptr; // Initialize static members
  // mutex Singleton::mtx;
  ```
</details>

<details>
  <summary>Factory</summary>

  - Factory Object: one factory for creating all objects
  ```cpp
  // Product Interface
  class Shape {
  public:
      virtual void draw() = 0;
      virtual ~Shape() = default;
  };
  
  // Concrete Product
  class Circle : public Shape {
  public:
      void draw() override {
          std::cout << "Drawing a Circle" << std::endl;
      }
  };
  class Square : public Shape {
  public:
      void draw() override {
          std::cout << "Drawing a Square" << std::endl;
      }
  };
  
  // Factory
  class ShapeFactory {
  public:
      static std::unique_ptr<Shape> createShape(const std::string& type) {
          if (type == "circle") return std::make_unique<Circle>();
          } else if (type == "square") return std::make_unique<Square>();
      }
  };
  
  int main() {
      auto shape1 = ShapeFactory::createShape("circle");
      shape1->draw();
      auto shape2 = ShapeFactory::createShape("square");
      shape2->draw();
      return 0;
  }
  ```

  - Factory Method Pattern: sub-factory classes define their own creations
  ```cpp
  // Interface Meat class
  class Meat {
  public:
      virtual void prepare() = 0;
      virtual ~Meat() {}
  };
  
  // Concrete classes
  class Beef : public Meat {
  public:
      void prepare() override {
          std::cout << "Preparing Beef!" << std::endl;
      }
  };
  class Chicken : public Meat {
  public:
      void prepare() override {
          std::cout << "Preparing Chicken!" << std::endl;
      }
  };
  
  // Interface Factory class
  class MeatFactory {
  public:
      virtual std::unique_ptr<Meat> createMeat() = 0;
      virtual ~MeatFactory() {}
  };
  
  // Concrete Factory
  class BeefFactory : public MeatFactory {
  public:
      std::unique_ptr<Meat> createMeat() override {
          return std::make_unique<Beef>();
      }
  };
  class ChickenFactory : public MeatFactory {
  public:
      std::unique_ptr<Meat> createMeat() override {
          return std::make_unique<Chicken>();
      }
  };
  
  // Client code
  int main() {
      std::unique_ptr<MeatFactory> beefFactory = std::make_unique<BeefFactory>();
      std::unique_ptr<Meat> beef = beefFactory->createMeat();
      beef->prepare();  // Output: Preparing Beef!
      std::unique_ptr<MeatFactory> chickenFactory = std::make_unique<ChickenFactory>();
      std::unique_ptr<Meat> chicken = chickenFactory->createMeat();
      chicken->prepare();  // Output: Preparing Chicken!
  
      return 0;
  }

  ```
</details>


Structural Pattern: to define relationship between objects
-
<details>
  <summary>Facade</summary>

  
</details>

<details>
  <summary>Adapter</summary>

  
</details>

<details>
  <summary>Composite</summary>

  
</details>

<details>
  <summary>Proxy</summary>

  
</details>

<details>
  <summary>Decorator</summary>

  
</details>

Behavioral Pattern: to define how objects behave and achieve the common goal
-
<details>
  <summary></summary>

  
</details>
<details>
  <summary></summary>

  
</details>
