## TypeScript Notes

TypeScript is widely adopted in the development community due to its static typing, which significantly enhances error detection and debugging during the development phase. Unlike JavaScript, where many errors only surface at runtime, TypeScript catches type-related mistakes early, reducing the likelihood of runtime errors. This feature not only increases code reliability but also boosts developer productivity by providing immediate feedback through integrated development environments (IDEs). Enhanced IDE support with features like autocompletion, intelligent code navigation, and robust refactoring tools further streamline the development process, making TypeScript an invaluable tool for creating maintainable and bug-free code.

Moreover, TypeScript's compatibility with modern JavaScript features and its support for future JavaScript proposals allow developers to write more expressive and cleaner code. By transpiling to JavaScript, TypeScript ensures that applications can run in any environment that supports JavaScript, without sacrificing the advantages of modern syntax and features. This future-proofing aspect, combined with TypeScript's ability to integrate seamlessly with popular frameworks and libraries like React, Angular, and Node.js, makes it particularly beneficial for large-scale projects. The strong typing system and modular code structure promote better collaboration within development teams, enabling scalable and consistent codebases that are easier to understand, extend, and maintain over time.

## Syntax

Simple types

```typescript
let age: number
let busy: boolean = true
const firstInput = document.querySelector('#first-input') as HTMLInputElement
function addNumbers(a: number, b: number) {
    screen.innerHTML = (a + b).toString()
}
```

Object with enum, tuple, type alias, and array of union types
```typescript
enum Permissions {
    ADMIN,
    READ_ONLY
}
type Age = 45 | 30 | 15
const you: {
    name: string;
    isReturning: boolean;
	permissions: Permissions; //enum type
	contact: [number, string]; //tuple type
    age: Age; //type alias
    stayedAt: (string | number)[]; //array type with union types
} = {
    name: 'Bobby',
    isReturning: true,
	permissions: Permissions.ADMIN, //enum
	contact: [+1123495082908, 'marywinkle@gmail.com'], //tuple
    age: 30, //alias
    stayedAt: ['florida-home', 'oman-flat', 'tokyo-bungalow', 23] //array
}
```


Function
```typescript
function add( firstValue: number, secondValue: number ) : number {
    return firstValue + secondValue
}
```


Array of objects
```typescript
const reviews : {
    name: string;
    stars: number;
    loyaltyUser: boolean;
    date: string;
}[] = [
    {
        name: 'Sheia',
        stars: 5,
        loyaltyUser: true,
        date: '01-04-2021'
    },
    {
        name: 'Andrzej',
        stars: 3,
        loyaltyUser: false,
        date: '28-03-2021'
    }//...
]
```

Interface to define structure of an object
```typescript
interface Review {
    name: string;
    stars: number;
    loyaltyUser: LoyaltyUser;
    date: string;
}
const reviews: Review[] = [
    {
        name: 'Sheia',
        stars: 5,
        loyaltyUser: LoyaltyUser.GOLD_USER,
        date: '01-04-2021'
    }]
```

Class to define blueprint of creating an object
```typescript
class Car {
    make: string
    year: number
    color: string
    constructor(make: string, year: number, color: string) {
        this.make = make
        this.year = year
        this.color = color
    }
}
```


## Tips
File organization is crucial in TypeScript projects for maintainability, scalability, and collaboration. It makes the codebase easier to navigate, allowing developers to locate and manage files efficiently, thus reducing errors during refactoring. Organized files support modular design, making projects scalable and enabling parallel development. This logical structure also facilitates better teamwork, as it helps all team members quickly understand and work with the project, promoting efficient collaboration and smoother onboarding for new developers.
```
	- enums.ts
	- types.ts
	- utils.ts    // for functions
	- interfaces.ts
	- classes.ts
```

Avoid using the `any` type in TypeScript, as it effectively disables type checking and defeats the purpose of using TypeScript's strong type system. Relying on `any` can lead to more runtime errors and reduced code reliability. Instead, use more specific types whenever possible to maintain type safety and clarity in your code. If you need a more flexible type that still provides some type safety, consider using `unknown` instead of `any`. This approach ensures that your code benefits from TypeScript's type-checking capabilities while allowing for flexibility where needed.