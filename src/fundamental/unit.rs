use std::cell::{Ref, RefCell};
use std::ops::{Add, Mul, Sub, Div};
use std::rc::{Rc, Weak};
use std::collections::HashMap;



#[derive(Clone)]
enum Operation {
    Add(UnitRef, UnitRef),
    //Sub,
    //Mul,
    //Div,
}

#[derive(Clone)]
struct _Unit {
    pub data: f32,
    pub grad: f32,
    operation: Option<Operation>,
}

impl PartialEq<Self> for _Unit {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }

    fn ne(&self, other: &Self) -> bool {
        self.eq(other)
    }
}



pub type Unit = Rc<RefCell<_Unit>>;
pub type UnitRef = Weak<RefCell<_Unit>>;

impl _Unit {

    pub fn new(data: f32) -> _Unit {
        _Unit {
            data,
            grad: 0.0,
            operation: None,
            //back_propagation_fn: None,
      //      children: None,
        }
    }

    pub fn to_unit(&self) -> Unit {
        Rc::new(RefCell::new(self.clone()))
    }

    pub fn to_unit_ref(&self) -> UnitRef {
        Rc::downgrade(&self.to_unit())
    }

    pub fn new_rc(data: f32) -> Unit {
        Rc::new(RefCell::new(_Unit {
            data,
            grad: 0.0,
            operation: None,
    //        children: None,
        }))
    }


    pub fn back_propagation(&self) {
        match self.operation {
            Some(Operation::Add(ref a, ref b)) => {
                let a = a.upgrade().unwrap();
                let b = b.upgrade().unwrap();
                a.borrow_mut().grad += self.grad;
                b.borrow_mut().grad += self.grad;
            }
            _ => {}
        }
    }

}


impl From<f32> for _Unit {
    fn from(data: f32) -> _Unit {
        _Unit::new(data)
    }
}

impl Add for _Unit {
    type Output = _Unit;

    fn add(self, other: _Unit) -> Self::Output {
        let mut output = _Unit::from(self.data + other.data);
      //  output.children = Some(vec![self.to_unit_ref(), other.to_unit_ref()]);
        output.operation = Some(Operation::Add(self.to_unit_ref(), other.to_unit_ref()));
        output
    }

}
//
// impl Sub for _Unit {
//     type Output = _Unit;
//
//     fn sub(self, other: _Unit) -> Self::Output {
//         _Unit::from(self.data - other.data)
//     }
// }
//
// impl Mul for _Unit {
//     type Output = _Unit;
//
//     fn mul(self, other: _Unit) -> Self::Output {
//         _Unit::from(self.data * other.data)
//     }
// }
//
// impl Div for _Unit {
//     type Output = _Unit;
//
//     fn div(self, other: _Unit) -> Self::Output {
//         _Unit::from(self.data / other.data)
//     }
// }
