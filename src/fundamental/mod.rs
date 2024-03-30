use std::cell::{Ref, RefCell};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::{Add, Deref, Div, Mul, Neg, Sub};
use std::rc::{Rc, Weak};

use op::{add, div, mul, pow, relu, sub, tanh};

pub mod autograd;
pub mod op;
pub mod unit;

use unit::_Unit;

#[derive(Clone, Eq, PartialEq, Debug)]
pub struct Unit(Rc<RefCell<_Unit>>);
//pub struct UnitRef(Weak<RefCell<_Unit>>);

#[derive(Clone, Debug)]
pub enum Operation {
    Add(Unit, Unit),
    Mul(Unit, Unit),
    Sub(Unit, Unit),
    Div(Unit, Unit),
    Tanh(Unit),
    ReLU(Unit),
    Pow(Unit, f32),
}

impl fmt::Display for Operation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Operation::Add(_, _) => write!(f, "Add"),
            Operation::Mul(_, _) => write!(f, "Mul"),
            Operation::Sub(_, _) => write!(f, "Sub"),
            Operation::Div(_, _) => write!(f, "Div"),
            Operation::Tanh(_) => write!(f, "Tanh"),
            Operation::ReLU(_) => write!(f, "ReLU"),
            Operation::Pow(_, _) => write!(f, "Pow"),
        }
    }
}

impl Unit {
    pub fn from<T>(t: T) -> Unit
    where
        T: Into<Unit>,
    {
        t.into()
    }

    fn new(u: _Unit) -> Unit {
        Unit(Rc::new(RefCell::new(u)))
    }

    pub fn data(&self) -> f32 {
        self.borrow().data
    }

    pub fn grad(&self) -> f32 {
        self.borrow().grad
    }

    pub fn zero_grad(&self) {
        self.borrow_mut().grad = 0.0;
    }

    pub fn adjust(&self, learning_rate: f32) {
        let mut u = self.borrow_mut();
        u.data += learning_rate * u.grad;
    }
}

impl Hash for Unit {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.borrow().hash(state)
    }
}

impl Deref for Unit {
    type Target = Rc<RefCell<_Unit>>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: Into<f32>> From<T> for Unit {
    fn from(data: T) -> Unit {
        Unit::new(_Unit::from(data.into()))
    }
}

impl Neg for Unit {
    type Output = Unit;

    fn neg(self) -> Self::Output {
        mul(&self, &Unit::from(-1.0))
    }
}

impl<'a> Neg for &'a Unit {
    type Output = Unit;

    fn neg(self) -> Self::Output {
        mul(self, &Unit::from(-1.0))
    }
}

impl Add<Unit> for Unit {
    type Output = Unit;

    fn add(self, other: Unit) -> Self::Output {
        add(&self, &other)
    }
}

impl<'a, 'b> Add<&'b Unit> for &'a Unit {
    type Output = Unit;

    fn add(self, other: &'b Unit) -> Self::Output {
        add(self, other)
    }
}

impl Sub<Unit> for Unit {
    type Output = Unit;

    fn sub(self, other: Unit) -> Self::Output {
        add(&self, &-other)
    }
}

impl<'a, 'b> Sub<&'b Unit> for &'a Unit {
    type Output = Unit;

    fn sub(self, other: &'b Unit) -> Self::Output {
        add(self, &-other)
    }
}

impl Mul<Unit> for Unit {
    type Output = Unit;

    fn mul(self, other: Unit) -> Self::Output {
        mul(&self, &other)
    }
}

impl<'a, 'b> Mul<&'b Unit> for &'a Unit {
    type Output = Unit;

    fn mul(self, other: &'b Unit) -> Self::Output {
        mul(self, other)
    }
}

impl Div<Unit> for Unit {
    type Output = Unit;

    fn div(self, other: Unit) -> Self::Output {
        div(&self, &other)
    }
}

impl<'a, 'b> Div<&'b Unit> for &'a Unit {
    type Output = Unit;

    fn div(self, other: &'b Unit) -> Self::Output {
        div(self, other)
    }
}
