Creational Pattern: for object creation
-
<details>
  <summary>Singleton</summary>
  
  - only one object accessible globally
  - Implementation: Singleton's constructor/destructor should always be private to prevent direct construction/desctruction calls with the `new`/`delete` operator
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
  - Implementation:
    - TypeInterface (for clients to interact with)
    - Type1, Type2 inherit TypeInterface for same behavior
    - TypeFactory contains method to create TypeInterface object
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
  - Implementation
    - TypeInterface for clients to interact with
    - Type1, Type2 inherit TypeInterface for same behavior
    - TypeFactory for clients to interact with
    - Type1Factory, Type2Factory inherit TypeFactory to create different types of object
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

  - Wrapper class to encapsulate subsytem while hiding details/complexities of the subsystem.
  - Key ideas: encapsulation, information hiding, separation of concerns
  - Subsystems should be private variables to hide the details (less coupling)
  - Implementation:
    - Type1, Type2 inherit TypeInterface for same behavior
    - Wrapper class knows TypeInterface and hides the interaction among them
  ```cpp
  class IAccount {
  public:
      virtual void deposit(double amount) = 0;
      virtual void withdraw(double amount) = 0;
      virtual ~IAccount() = default;
  };
  
  class CheckingAccount : public IAccount {
      void deposit(double amount) override {
          std::cout << "Deposited $" << amount << " into Checking Account.\n";
      }
      void withdraw(double amount) override {
          std::cout << "Withdrew $" << amount << " from Checking Account.\n";
      }
  };
  //class SavingsAccount : public IAccount {...}
  
  class BankService {
  private:
      unordered_map<int, unique_ptr<IAccount>> bankAccounts; // Map of account ID to account object
      int nextAccountId = 1;
  
  public:
      int createNewAccount(const string& type, int initialAmount) {
          unique_ptr<IAccount> account; // Create the appropriate account type based on the input
          if (type == "Checking") account = make_unique<CheckingAccount>();
          else if (type == "Savings") account = make_unique<SavingsAccount>();
          else if (type == "Investment") account = make_unique<InvestmentAccount>();
  
          account->deposit(initialAmount); // Initialize the account with the initial deposit
          int accountId = nextAccountId++; 
          bankAccounts[accountId] = move(account); // Assign the account an ID and store it in the map
          return accountId;
      }
  
      void transferFromAccountToAccount(int fromId, int toId, double amount) {
          if (bankAccounts.find(fromId) != bankAccounts.end() and bankAccounts.find(toId) != bankAccounts.end()) {
              bankAccounts[fromId]->withdraw(amount);
              bankAccounts[toId]->deposit(amount);
          }
      }
  };
  
  int main() {
      BankService bankService;
      int account1 = bankService.createNewAccount("Checking", 1000);
      int account2 = bankService.createNewAccount("Savings", 2000);
      bankService.transferFromAccountToAccount(account1, account2, 500); // Deposit to accounts
      return 0;
  }
  ```
</details>

<details>
  <summary>Adapter</summary>

  - Provides abstraction interface of the third-party classes for the clients to interact with.
  - Eliminates the risk of breaking subsystem (target) while not changing the third-party library (adaptee)
  - Implementation:
    - TargetInterface for clients to interact with
    - Adaptee class for incompatible behaviors
    - Adapter inherit TargetInterface

  ```cpp
  // Target class
  class CoffeMachineInterface{
  public:
      CoffeMachineInterface(){};
      virtual void chooseFirstSelection() = 0;
      virtual void chooseSecondSelection() = 0;
      
  };

  // Adaptee class
  class OldCoffeeMachine{
  public:
      void selectA(){
          cout << "Old machine A selected" <<endl;
      }
      void selectB(){
          cout << "Old machine B selected" <<endl;
      }
  };
  
  class CoffeeTouchscreenAdapter: public CoffeMachineInterface{
  private:
      OldCoffeeMachine* oldMachine;
  public:
      void connect(OldCoffeeMachine* om){
          oldMachine = om;
      }
      void chooseFirstSelection() override{
          oldMachine->selectA();
      }
      void chooseSecondSelection() override{
          oldMachine->selectB();
      }
  };
  
  int main(){
      OldCoffeeMachine ocm;
      CoffeeTouchscreenAdapter adapter;
      adapter.connect(&ocm);
      adapter.chooseFirstSelection();
      return 0;
  }
  ```
</details>

<details>
  <summary>Composite</summary>

  - Deals with nested objects/structures by enforcing polymorphism and building a tree-like structure.
  - Leaf & composite both inheritate from the same interface while composite can grow the tree and leaf ends the tree
  - Implementation:
    - CompositeInterface
    - Leaf (1 CompositeInterface), CompositeObject (mulitple instances of CompositeInterface) inherit CompositeInterface
  ```cpp
  // Component interface (base class for all shapes)
  class Graphic {
  public:
      virtual void draw() const = 0; // Pure virtual method
      virtual ~Graphic() = default;  // Virtual destructor
  };
  
  // Leaf class (simple shapes like Circle and Rectangle)
  class Circle : public Graphic {
  public:
      void draw() const override {
          cout << "Drawing a Circle\n";
      }
  };
  
  // Composite class (a group of shapes)
  class CompositeGraphic : public Graphic {
  private:
      vector<Graphic*> children; // List of child graphics
  
  public:
      ~CompositeGraphic() {
          for (auto child : children) {
              delete child; // Ensure proper cleanup
          }
      }
  
      void add(Graphic* graphic) {
          children.push_back(graphic);
      }
  
      void remove(Graphic* graphic) {
          children.erase(remove(children.begin(), children.end(), graphic), children.end());
      }
  
      void draw() const override {
          cout << "Drawing a CompositeGraphic containing:\n";
          for (const auto& child : children) {
              child->draw();
          }
      }
  };
  
  // Client code
  int main() {
      // Create simple shapes
      Circle* circle1 = new Circle();
      Circle* circle2 = new Circle();
      // Create a composite graphic
      CompositeGraphic* group = new CompositeGraphic();
      group->add(circle1);
      // Create another composite group and nest it
      CompositeGraphic* nestedGroup = new CompositeGraphic();
      nestedGroup->add(circle2);
      nestedGroup->add(group);
      // Draw everything
      nestedGroup->draw();
      // Clean up
      delete nestedGroup; // This also deletes `group`, `circle1`, `circle2`, and `rectangle`.
      return 0;
  }
  ```
</details>

<details>
  <summary>Proxy</summary>

  - Represents a simplified, lighter version of the original object and Behaves the same but may request the action of original object
  - Purpose: smaller proxy (saves space when original object is too large), protection proxy (sensitive data in original one or role-based access control), remote proxy (real one exists in Cloud and you work on virtual one to update periodically).
  - Implementation:
    - ObjectInterface
    - Proxy (1 lazy reference to Object), Object inherit ObjectInterface

  ```cpp
// Subject Interface (common interface for RealSubject and Proxy)
class Image {
public:
    virtual void display() const = 0; // Interface method
    virtual ~Image() = default;
};

// RealSubject class (heavy object)
class HighResolutionImage : public Image {
private:
    string filename;
    void loadFromDisk() const {
        cout << "Loading high-resolution image from disk: " << filename << endl;
    }

public:
    HighResolutionImage(const string& file) : filename(file) {
        loadFromDisk(); // Simulate expensive operation
    }
    void display() const override {
        cout << "Displaying high-resolution image: " << filename << endl;
    }
};

// Proxy class
class ImageProxy : public Image {
private:
    string filename;
    mutable HighResolutionImage* realImage; // Lazy-loaded

public:
    ImageProxy(const string& file) : filename(file), realImage(nullptr) {}
    ~ImageProxy() {
        delete realImage; // Ensure proper cleanup
    }
    void display() const override {
        if (!realImage) {
            realImage = new HighResolutionImage(filename); // Load image lazily
        }
        realImage->display();
    }
};

// Client code
int main() {
    // Client uses the Proxy instead of directly using the real object
    Image* image = new ImageProxy("example.jpg");
    image->display(); // Image is loaded and displayed only when needed
    delete image; // Cleanup
    return 0;
}
```
</details>

<details>
  <summary>Decorator</summary>

  - Attaches a stack of behaviors to an object by adding a "has-a" relationship via aggregations
  - Implementation:
    - BasicObjectInterface (for clients to interact with)
    - BasicObject and DecoratorInterface implements BasicObjectInterface (is a type of)
    - DecoratorA, DecoratorB ... implements DecoratorInterface

  ```cpp
  // Base interface for Coffee
  class Coffee {
  public:
      virtual ~Coffee() {}
      virtual double cost() const = 0;
  };
  
  // Concrete implementation of the Basic Coffee class
  class SimpleCoffee : public Coffee {
  public:
      double cost() const override {
          return 5.0;
      }
  };
  
  // Interface for CoffeeDecorator
  class CoffeeDecorator : public Coffee {
  public:
      virtual ~CoffeeDecorator() {}
  };
  
  // Concrete decorator: Milk
  class MilkDecorator : public CoffeeDecorator {
  private:
      Coffee* coffee;
  public:
      MilkDecorator(Coffee* coffee) : coffee(coffee) {}
      double cost() const override {
          return coffee->cost() + 1.5; // Adding cost for milk
      }
  };
  
  
  // Main function to demonstrate the decorator pattern
  int main() {
      // Create a simple coffee
      Coffee* myCoffee = new SimpleCoffee();
      myCoffee = new MilkDecorator(myCoffee); // Add milk to the coffee
      delete myCoffee; // Clean up
      return 0;
  }
  ```
</details>

Behavioral Pattern: to define how objects collaborate and achieve the common goal
-
<details>
  <summary>Template Method</summary>

  - Template class contains common steps while derived class contains special steps
  - Implementation:
    - TemplateAbstractClass has virtual methods for special steps and concrete methods for common steps (including `virtual` gives derived class the ability to override)
    - ConcreteObject inherits TemplateAbstractClass and overrides special steps 
  ```cpp
  // Abstract Base Class
  class PastaDish {
  public:
      // Template Method
      void makeRecipe()  {
          boilWater();
          addPasta();
          addSauce();
          addProtein();
      }
  
  protected:
      virtual void addPasta() = 0; // Abstract methods to be implemented by subclasses
      virtual void addSauce() = 0;
      virtual void addProtein() = 0;
  
  private:
      void boilWater() { // Common step
          std::cout << "Boiling water.\n";
      }
  };
  
  // Concrete Subclass: Spaghetti with Meatballs
  class SpaghettiMeatballs : public PastaDish {
  protected:
      void addPasta() override {
          std::cout << "Adding spaghetti noodles.\n";
      }
  
      void addSauce() override {
          std::cout << "Adding tomato sauce.\n";
      }
  
      void addProtein() override {
          std::cout << "Adding meatballs.\n";
      }
  
  };
  
  // Concrete Subclass: Penne Alfredo
  class PenneAlfredo : public PastaDish {
  protected:
      void addPasta() override {
          std::cout << "Adding penne noodles.\n";
      }
  
      void addSauce() override {
          std::cout << "Adding Alfredo sauce.\n";
      }
  
      void addProtein() override {
          std::cout << "Adding grilled chicken.\n";
      }
  
  };
  
  // Main Function
  int main() {
      SpaghettiMeatballs spaghettiDish;
      PenneAlfredo penneDish;
  
      std::cout << "Making Spaghetti with Meatballs:\n";
      spaghettiDish.makeRecipe();
  
      std::cout << "\nMaking Penne Alfredo:\n";
      penneDish.makeRecipe();
  
      return 0;
  }
  ```
</details>

<details>
  <summary>Chain of Responsibility</summary>

  - Requests are handled/tried with different handlers until we succeed or run out of handlers
  - Purpose: multi-filters
  - Implementation:
    - AbstractHandler with template steps (if fails, call next handler)
    - ConcreteHandler with special steps (check if rules matches. If matches, do something)
  ```cpp
  // Abstract Base Class for Handlers
  class SupportHandler {
  protected:
      SupportHandler* nextHandler = nullptr; // Pointer to the next handler in the chain
  
  public:
      void setNextHandler(SupportHandler* handler) {
          nextHandler = handler;
      }
      void handleRequest(const std::string& issue){
          bool handled = handling(issue);
          if(handled) return;
          if (nextHandler) nextHandler->handleRequest(issue);
          else std::cout << "Frontline Support: Unable to handle the request.\n";
      };
      virtual bool handling(const std::string& issue) = 0;
  };
  
  // Concrete Handler: Frontline Support
  class FrontlineSupport : public SupportHandler {
  public:
      bool handling(const std::string& issue) override {
          if (issue == "basic") {
              std::cout << "Frontline Support: Handled the basic issue.\n";
              return true;
          }
          return false;
      }
  };
  
  // Concrete Handler: Technical Support
  class TechnicalSupport : public SupportHandler {
  public:
      bool handling(const std::string& issue) override {
          if (issue == "technical") {
              std::cout << "Technical Support: Handled the technical issue.\n";
              return true;
          }
          return false;
      }
  };
  
  // Concrete Handler: Manager Support
  class ManagerSupport : public SupportHandler {
  public:
      bool handling(const std::string& issue) override {
          if (issue == "management") {
              std::cout << "Manager Support: Handled the management issue.\n";
              return true;
          }
          return false;
      }
  };
  
  // Main Function
  int main() {
      // Handlers
      FrontlineSupport frontline;
      TechnicalSupport technical;
      ManagerSupport manager;
  
      // Setting up the chain
      frontline.setNextHandler(&technical);
      technical.setNextHandler(&manager);
  
      // Test cases
      std::cout << "Sending 'basic' request:\n";
      frontline.handleRequest("basic");
  
      std::cout << "\nSending 'management' request:\n";
      frontline.handleRequest("management");
  
      std::cout << "\nSending 'unknown' request:\n";
      frontline.handleRequest("unknown");
      return 0;
  }
  
  ```
</details>

<details>
  <summary>State</summary>

  - Used when behavior changes if state changes
  - Implementation:
    - StateInterface has common virtual methods (behaviors)
    - Object class has the following
      - constructor placeholder (to be implemented later)
      - state objects with getters
      - same behavior methods. Each calls current_state's virtual behavior method
    - ConcreteState inherits StateInterface and overrides virtual behavior methods
    - Object constructor is implemented
  ```cpp
  // Forward declaration of VendingMachine
  class VendingMachine;
  
  // State Interface
  class State {
  public:
      virtual void insertDollar(VendingMachine* vendingMachine) = 0;
      virtual void ejectMoney(VendingMachine* vendingMachine) = 0;
      virtual void dispense(VendingMachine* vendingMachine) = 0;
      virtual ~State() = default;
  };
  
  // VendingMachine Class
  class VendingMachine {
  private:
      State* idleState;
      State* hasOneDollarState;
      State* outOfStockState;
  
      State* currentState;
      int stock;
  
  public:
      VendingMachine(int count); //constructor implemented later because concrete states are not created yet
  
      void setState(State* state) { currentState = state; }
      State* getIdleState() { return idleState; }
      State* getHasOneDollarState() { return hasOneDollarState; }
      State* getOutOfStockState() { return outOfStockState; }
  
      void insertDollar() { currentState->insertDollar(this); }
      void ejectMoney() { currentState->ejectMoney(this); }
      void dispense() { currentState->dispense(this); }
  
      void releaseProduct() {
          if (stock > 0) {
              stock--;
              cout << "Product dispensed. Remaining stock: " << stock << "\n";
          }
      }
  
      int getStock() const { return stock; }
  };
  
  // IdleState Class
  class IdleState : public State {
  public:
      void insertDollar(VendingMachine* vendingMachine) override {
          cout << "Dollar inserted.\n";
          vendingMachine->setState(vendingMachine->getHasOneDollarState());
      }
  
      void ejectMoney(VendingMachine* vendingMachine) override {
          cout << "No money to return. Machine is idle.\n";
      }
  
      void dispense(VendingMachine* vendingMachine) override {
          cout << "Payment required before dispensing.\n";
      }
  };
  
  // HasOneDollarState Class
  class HasOneDollarState : public State {
  public:
      void insertDollar(VendingMachine* vendingMachine) override {
          cout << "Already have one dollar.\n";
      }
  
      void ejectMoney(VendingMachine* vendingMachine) override {
          cout << "Returning money.\n";
          vendingMachine->setState(vendingMachine->getIdleState());
      }
  
      void dispense(VendingMachine* vendingMachine) override {
          if (vendingMachine->getStock() > 1) {
              vendingMachine->releaseProduct();
              vendingMachine->setState(vendingMachine->getIdleState());
          } else {
              vendingMachine->releaseProduct();
              vendingMachine->setState(vendingMachine->getOutOfStockState());
          }
      }
  };
  
  // OutOfStockState Class
  class OutOfStockState : public State {
  public:
      void insertDollar(VendingMachine* vendingMachine) override {
          cout << "Machine is out of stock. Returning your dollar.\n";
      }
  
      void ejectMoney(VendingMachine* vendingMachine) override {
          cout << "No money to return. Machine is out of stock.\n";
      }
  
      void dispense(VendingMachine* vendingMachine) override {
          cout << "Cannot dispense. Machine is out of stock.\n";
      }
  };
  
  // Implementation of VendingMachine Constructor
  VendingMachine::VendingMachine(int count) : stock(count) {
      idleState = new IdleState();
      hasOneDollarState = new HasOneDollarState();
      outOfStockState = new OutOfStockState();
  
      currentState = (stock > 0) ? idleState : outOfStockState;
  }
  
  // Main Function
  int main() {
      VendingMachine machine(2); // Initialize vending machine with 2 items in stock
  
      cout << "--- Test Case 1: Insert dollar and dispense product ---\n";
      machine.insertDollar();
      machine.dispense();
  
      cout << "\n--- Test Case 2: Try to dispense without inserting money ---\n";
      machine.dispense();
  
      cout << "\n--- Test Case 3: Eject money ---\n";
      machine.insertDollar();
      machine.ejectMoney();
  
      cout << "\n--- Test Case 4: Out of stock ---\n";
      machine.insertDollar();
      machine.dispense(); // Dispense last product
      machine.insertDollar(); // Try to buy when out of stock
  
      return 0;
  }
  ```
</details>

<details>
  <summary>Command</summary>

  - Schedule tasks, redo/undo
  - Implementation:
    - CommandInterface
    - ConcreteCommand inherits CommandInterface and keeps a reference to Receiver
    - CommandManager keeps track of undo/redo list of ConcreteCommand
    - Receiver has actions for ConcreteCommand to call.
  ```cpp
  // Abstract Command class
  class Command {
  public:
      virtual ~Command() = default;
      virtual void execute() = 0;
      virtual void unexecute() = 0;
      virtual bool isReversible() const = 0;
  };
  
  // Receiver class
  class Document {
  public:
      void insertText(const std::string& text, size_t position) {
          content.insert(position, text);
          std::cout << "Inserted text: \"" << text << "\" at position " << position << std::endl;
      }
  
      void deleteText(size_t position, size_t length) {
          if (position + length <= content.size()) {
              content.erase(position, length);
              std::cout << "Deleted text of length " << length << " from position " << position << std::endl;
          }
      }
  
      void display() const {
          std::cout << "Document content: \"" << content << "\"" << std::endl;
      }
  
  private:
      std::string content;
  };
  
  // Concrete Command for "Paste" operation
  class PasteCommand : public Command {
  public:
      PasteCommand(Document& doc, const std::string& text, size_t position)
          : document(doc), textToInsert(text), position(position) {}
  
      void execute() override {
          document.insertText(textToInsert, position);
          executed = true;
      }
  
      void unexecute() override {
          if (executed) {
              document.deleteText(position, textToInsert.length());
          }
      }
  
      bool isReversible() const override {
          return true;
      }
  
  private:
      Document& document;
      std::string textToInsert;
      size_t position;
      bool executed = false;
  };
  
  // Invoker (Command Manager)
  class CommandManager {
  public:
      ~CommandManager() {
          clearHistory();
      }
  
      void executeCommand(Command* command) {
          if (command->isReversible()) {
              command->execute();
              history.push(command);
          } else {
              command->execute();
              delete command;
          }
      }
  
      void undo() {
          if (!history.empty()) {
              Command* command = history.top();
              command->unexecute();
              delete command;
              history.pop();
          } else {
              std::cout << "No commands to undo!" << std::endl;
          }
      }
  
  private:
      void clearHistory() {
          while (!history.empty()) {
              delete history.top();
              history.pop();
          }
      }
  
      std::stack<Command*> history;
  };
  
  // Client
  int main() {
      Document doc;
      CommandManager commandManager;
      // Paste operation
      Command* pasteCommand1 = new PasteCommand(doc, "Hello", 0);
      commandManager.executeCommand(pasteCommand1);
      // Paste another text
      Command* pasteCommand2 = new PasteCommand(doc, " World", 5);
      commandManager.executeCommand(pasteCommand2);
      doc.display();
      // Undo the last command
      commandManager.undo();
      doc.display();
      commandManager.undo();
      doc.display();
  
      return 0;
  }
  ```
</details>

<details>
  <summary>Mediator</summary>

  - Advantages: Loose coupling between colleagues allows for easier reuse and maintainability, with centralized interaction logic improving readability and extensibility.
  - Disadvantages: The mediator can become overly large and complex, making it harder to debug and potentially undermining the benefits of centralization.
  - Implementation:
    - Mediator and Colleague interfaces
    - ConcreteColleague1, ConcreteColleague2 inherit Colleague
    - ConcreteMediator inherits Mediator and know ConcreteColleague1, ConcreteColleague2
  ```cpp
  // Forward declarations
  class HouseMediator;
  
  // Abstract Colleague class
  class Colleague {
  protected:
      HouseMediator* mediator;
  public:
      Colleague(HouseMediator* mediator) : mediator(mediator) {}
      virtual ~Colleague() {}
      virtual void notify(const std::string& event) = 0;
  };
  
  // Mediator Interface
  class HouseMediator {
  public:
      virtual ~HouseMediator() {}
      virtual void notify(Colleague* sender, const std::string& event) = 0;
  };
  
  // Concrete Colleague classes
  class Phone : public Colleague {
  public:
      Phone(HouseMediator* mediator) : Colleague(mediator) {}
      void alarmGoesOff() {
          std::cout << "Phone: Alarm is going off.\n";
          mediator->notify(this, "Alarm");
      }
      void notify(const std::string& event) override {}
  };
  
  class CoffeeMaker : public Colleague {
  public:
      CoffeeMaker(HouseMediator* mediator) : Colleague(mediator) {}
      void brewCoffee() {
          std::cout << "CoffeeMaker: Brewing coffee.\n";
      }
      void notify(const std::string& event) override {}
  };
  
  class Tablet : public Colleague {
  public:
      Tablet(HouseMediator* mediator) : Colleague(mediator) {}
      void loadNewspaper() {
          std::cout << "Tablet: Loading the latest Globe and Mail.\n";
      }
      void notify(const std::string& event) override {}
  };
  
  // Concrete Mediator
  class ConcreteHouseMediator : public HouseMediator {
  private:
      Phone* phone;
      CoffeeMaker* coffeeMaker;
      Tablet* tablet;
  
  public:
      void setPhone(Phone* phone) { this->phone = phone; }
      void setCoffeeMaker(CoffeeMaker* coffeeMaker) { this->coffeeMaker = coffeeMaker; }
      void setTablet(Tablet* tablet) { this->tablet = tablet; }
  
      void notify(Colleague* sender, const std::string& event) override {
          if (event == "Alarm") {
              std::cout << "Mediator: Handling 'Alarm' event.\n";
              coffeeMaker->brewCoffee();
              tablet->loadNewspaper();
          }
      }
  };
  
  // Main function
  int main() {
      ConcreteHouseMediator mediator;
  
      Phone phone(&mediator);
      CoffeeMaker coffeeMaker(&mediator);
      Tablet tablet(&mediator);
  
      mediator.setPhone(&phone);
      mediator.setCoffeeMaker(&coffeeMaker);
      mediator.setTablet(&tablet);
  
      // Simulate alarm going off
      phone.alarmGoesOff();
  
      return 0;
  }
  ```
</details>

<details>
  <summary>Observer</summary>

  - Implementation:
    - Subscriber inherits Observer and gets the notification if new changes to a post
    - Post inherits Subject and keeps a list of Observers to notify

  ```cpp
  // Observer Interface
  class Observer {
  public:
      virtual ~Observer() {}
      virtual void update(const std::string& blogPost) = 0; // Notify observer of a change
  };
  
  // Subject Base Class
  class Subject {
  protected:
      std::vector<Observer*> observers;
  
  public:
      virtual ~Subject() {}
  
      void registerObserver(Observer* observer) {
          observers.push_back(observer);
      }
  
      void unregisterObserver(Observer* observer) {
          observers.erase(std::remove(observers.begin(), observers.end(), observer), observers.end());
      }
  
      void notifyObservers(const std::string& blogPost) {
          for (Observer* observer : observers) observer->update(blogPost);
      }
  };
  
  // Concrete Subject (Blog)
  class Blog : public Subject {
  private:
      std::string latestPost;
  
  public:
      void addPost(const std::string& post) {
          latestPost = post;
          notifyObservers(latestPost); // Notify all subscribers of the new post
      }
  };
  
  // Concrete Observer (Subscriber)
  class Subscriber : public Observer {
  private:
      std::string name;
  
  public:
      Subscriber(const std::string& name) : name(name) {}
  
      void update(const std::string& blogPost) override {
          std::cout << "Subscriber " << name << " received notification: New blog post -> " << blogPost << "\n";
      }
  };
  
  // Main Function
  int main() {
      // Create blog
      Blog blog;
  
      // Create subscribers
      Subscriber subscriber1("Alice");
      Subscriber subscriber2("Bob");
      Subscriber subscriber3("Charlie");
  
      // Register subscribers to the blog
      blog.registerObserver(&subscriber1);
      blog.registerObserver(&subscriber2);
      blog.registerObserver(&subscriber3);
  
      // Add a new post
      blog.addPost("Observer Pattern in C++");
  
      // Unregister one subscriber
      blog.unregisterObserver(&subscriber2);
  
      // Add another post
      blog.addPost("Understanding Design Patterns");
  
      return 0;
  }
  ```
</details>


MVC
-
<details>
  <summary>Model</summary>
  
  ```cpp
  class StoreOrder {
  private:
      std::vector<std::pair<std::string, double>> items; // Item name and price
      std::vector<IObserver*> observers; // Observers (views)
  
  public:
      // Add an observer (view)
      void addObserver(IObserver* observer) {
          observers.push_back(observer);
      }
  
      // Notify all observers when the model changes
      void notifyObservers() {
          for (IObserver* observer : observers) {
              observer->update();
          }
      }
  
      // Add an item to the order
      void addItem(const std::string& name, double price) {
          items.push_back({name, price});
          notifyObservers();
      }
  
      // Remove an item from the order
      void removeItem(const std::string& name) {
          items.erase(std::remove_if(items.begin(), items.end(), 
              [&name](const std::pair<std::string, double>& item) {
                  return item.first == name;
              }), items.end());
          notifyObservers();
      }
  
      // Change the price of an item
      void changePrice(const std::string& name, double newPrice) {
          for (auto& item : items) {
              if (item.first == name) {
                  item.second = newPrice;
                  break;
              }
          }
          notifyObservers();
      }
  
      // Get the list of items
      const std::vector<std::pair<std::string, double>>& getItems() const {
          return items;
      }
  };
  ```
</details>

<details>
  <summary>View: keep track of Model and update if needed</summary>

  ```cpp
  class IObserver {
  public:
      virtual void update() = 0; // The update method for observers (views)
  };
  
  class OrderView : public IObserver {
  private:
      StoreOrder& storeOrder;
  
  public:
      OrderView(StoreOrder& order) : storeOrder(order) {
          storeOrder.addObserver(this);
      }
  
      // Display the items in the order
      void display() {
          std::cout << "Items in the order:\n";
          const auto& items = storeOrder.getItems();
          for (const auto& item : items) {
              std::cout << item.first << " - $" << item.second << std::endl;
          }
          std::cout << "Total: $" << calculateTotal() << "\n\n";
      }
  
      // Calculate the total price of the order
      double calculateTotal() {
          double total = 0.0;
          for (const auto& item : storeOrder.getItems()) {
              total += item.second;
          }
          return total;
      }
  
      // This method is called when the model (StoreOrder) updates
      void update() override {
          display(); // Update the view when the model changes
      }
  };
  ```
</details>

<details>
  <summary>Controller: keep track of Model and View to manipulate Model based on input from View</summary>

  ```cpp
  class Controller {
  private:
      StoreOrder& storeOrder;
      OrderView& orderView;
  
  public:
      Controller(StoreOrder& order, OrderView& view) : storeOrder(order), orderView(view) {}
  
      // Add an item to the order
      void addItemToOrder(const std::string& name, double price) {
          storeOrder.addItem(name, price);
      }
  
      // Remove an item from the order
      void removeItemFromOrder(const std::string& name) {
          storeOrder.removeItem(name);
      }
  
      // Change the price of an item
      void changeItemPrice(const std::string& name, double newPrice) {
          storeOrder.changePrice(name, newPrice);
      }
  };
  ```
</details>

Design Principles Underlying Design Patterns
-

Liskow Substitution: S inheirs B. Then we should be able to replace all B with S without changing the behaviors of a program.
Open/Closed: Open to extension (good practice for backward compatibility) but closed to change. Stable parts should be closed. Unstable parts should be extensible.


Dependency Inversion: high-level modules should depend on high-level generalizations (not the low-level details). Client classes should call methods in abstract classes (generalizations). Decrease coupling (high coupling = hard to change) by changing direct calls to indirect calls

Composing Object: code reuse should be achieved via aggregation instead of inheritance. (Include hpp abstract class instead cpp concrete class when reusing).

- Inheritance: "is-a". highly coupled
- Aggregation: "has-a". object in another object. Lifetime is the same
- Composition: "has-a". object reference in another object. Lifetime can be different (Object reference can live longer). Adds behaviors at runtime

```cpp
Person::Person(string name, Birthday dateOfBirth):name(name), dateOfBirth(dateOfBirth) {
  //this->name = name;
  //this->dateOfBirth = dateOfBirth; //This is not allowed when initializing reference/non-basic objects!
}
```

Interface Segregation: object A has union of two different object's behaviors (B & C), we should separate and implement two interfaces.
- Bad design: B implemnets CInterface. A inherits B (highly coupled)
- Better design: A implements BInterface, CInterface
```cpp
class ICashier{
public:
    // ICashier();
    virtual void addItem() = 0;
    virtual void checkout() = 0;
};
class IWorker{
public:
    // IWorker();
    virtual void rest() = 0;
    virtual void startShift() = 0;
};

class MachineCashier: ICashier{
public:
    MachineCashier(){
        
    }
    void addItem() override{
        cout << "AutoCashier adds item" << endl;
    }
    void checkout() override{
        cout << "AutoCashier checks out" << endl;
    }
};

class HumanCashier: ICashier, IWorker{
public:
    HumanCashier(){
        
    }
    void addItem() override{
        cout << "HumanCashier adds item" << endl;
    }
    void checkout() override{
        cout << "HumanCashier checks out" << endl;
    }
    void rest() override{
        cout << "HumanCashier rests" << endl;
    }
    
    void startShift() override{
        cout << "HumanCashier starts" << endl;
    }
};

// Main Function
int main() {
    HumanCashier hc;
    hc.startShift();
    MachineCashier mc;
    mc.addItem();
    return 0;
}
```


Least Knowledge
- A method can call other methods within the same class.
- A method can call methods of parameters passed to it.
- A method can call methods of objects it instantiates locally (within the method).
  - If class A doesn't contain class C as a parameter or field inside A, that method should NOT call C's methods. 
- A method can call methods of instance variables (direct components) of its class.


Anti-Patterns (Code Smells)
- Too few comments: Makes code hard to understand for others or yourself.
- Too many comments: Can indicate bad design and may become outdated as code changes.
- Duplicated Code needs to be refactored based on D.R.Y. Principle (Don’t Repeat Yourself)
- Long Methods may indicate excessive complexity.
- Large Classes (God Classes) accumulate too many responsibilities(later may attract more responsibilities) and become difficult to maintain => Break down large classes into more cohesive
- Small Classes (Data Classes): Data-only classes with no functionality (e.g., a class with only getters/setters) are not ideal => Consider adding relevant methods to the class to make it a better abstraction.
- Data Clumps: groups of related data (e.g., x, y, z coordinates) should be encapsulated in a single object (e.g., a 3D point class).
- Long Parameter Lists: methods with many parameters are harder to use and more prone to errors => use parameter objects to group related parameters into a single object
- Divergent Change requires multiple changes of a class for different reasons => Break the class into smaller, cohesive classes with distinct responsibilities
- Shotgun Surgery: A single change requires modifications in multiple classes => Consolidate related methods into fewer classes, but avoid creating large classes (to prevent God classes).
- Feature Envy happens (if two class always talk to each other, maybe they should be together) when a method is more interested in the details of another class than its own => Move the method to the class it is more dependent on to improve cohesion
- Inappropriate Intimacy occurs when two classes are tightly coupled, constantly interacting with each other => Refactor to break the dependency cycle, potentially by extracting common methods into a third class.
- Message Chains violates the Law of Demeter, where you call a method on an object that returns another object, leading to a chain of calls => Reduce long message chains to improve design flexibility and reduce brittleness.
- Primitive Obsession: Excessive use of primitive types (like int, float, or String) instead of creating appropriate abstractions => Define custom classes to represent domain-specific data (e.g., PostalCode class instead of using a string).
- Switch Statements => consider using polymorphism to handle different cases instead of relying on switch statements.
- Speculative Generality: Over-engineering by adding unnecessary generalization (like unused superclasses or interfaces) in anticipation of future requirements => Follow "just-in-time design" by keeping things simple and only generalizing when needed.
- Refused Request: A subclass inherits methods it doesn’t need, leading to unused or irrelevant behavior => If the subclass doesn't need the functionality, don’t inherit it. Instead, define the behavior directly in the subclass or elsewhere.

Notes:
- A "owns" B = Composition : B has no meaning or purpose in the system without A
- A "uses" B = Aggregation : B exists independently (conceptually) from A
