use std::rc::Rc;
use std::cell::RefCell;

#[derive(Debug)]
struct ValueInternal {
    data: f64,
    grad: f64,
    _prev: Vec<Value>,
    _op: String,
}

#[derive(Clone, Debug)]
pub struct Value(Rc<RefCell<ValueInternal>>);

impl Value {
    pub fn new(data: f64) -> Self{
        Value(Rc::new(RefCell::new(ValueInternal {
            data,
            grad: 0.0,
            _prev: Vec::new(),
            _op: String::new(),
        })))
    }
    pub fn data(&self) -> f64 { self.0.borrow().data }
    pub fn grad(&self) -> f64 { self.0.borrow().grad }
    pub fn set_grad(&self, val: f64) { self.0.borrow_mut().grad = val; }
    pub fn add_grad(&self, val: f64) { self.0.borrow_mut().grad += val; }
    pub fn zero_grad(&self){
        self.set_grad(0.0);
    }
    pub fn mul(&self, other: &Value) -> Value {
        Value(Rc::new(RefCell::new(ValueInternal {
            data: self.data() * other.data(),
            grad: 0.0,
            _prev: vec![self.clone(), other.clone()],
            _op: "*".to_string(),
        })))
    }
    pub fn add(&self, other: &Value) -> Value {
        Value(Rc::new(RefCell::new(ValueInternal {
            data: self.data() + other.data(),
            grad: 0.0,
            _prev: vec![self.clone(), other.clone()],
            _op: "+".to_string(),
        })))
    }
    // pub fn add(self: &Rc<Self>, other: &Rc<Self>) -> Rc<Value> {
    //     Rc::new(Value {
    //         data: self.data + other.data,
    //         grad: 0.0,
    //         _prev: vec![Rc::clone(self), Rc::clone(other)],
    //         _op: "+".to_string(),
    //     })
    // }
}

fn main() {
    let v = Value::new(3.0);
    println!("Value: {}, Gradient: {}", v.data(), v.grad());
    v.set_grad(1.5); // Simulate a gradient computation
    println!("Value: {}, Gradient: {}", v.data(), v.grad());
    v.zero_grad();
    println!("After zero_grad - Value: {}, Gradient: {}", v.data(), v.grad());

    let a = Value::new(2.0);
    let b = Value::new(5.0);
    let c = a.mul(&b);
    println!("Result of multiplication a * b: {}", c.data());

    let d = a.add(&b);
    println!("Result of addition a + b: {}", d.data());
}
