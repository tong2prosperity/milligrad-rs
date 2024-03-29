use super::*;
use super::unit::{new_unit, Operation};
use super::Unit;



pub fn add(l: &Unit, r: &Unit) -> Unit {
    let result = l.borrow().data + r.borrow().data;
    let (l_rc, r_rc) = (Rc::downgrade(l), Rc::downgrade(r));
    let op = Some(Operation::Add(l_rc, r_rc));
    Unit::new(_Unit::new(
        result,
        op,
        vec![l.clone(), r.clone()]
    ))
}

pub fn mul(l: &Unit, r: &Unit) -> Unit {
    let result = l.borrow().data * r.borrow().data;
    let (l_rc, r_rc) = (Rc::downgrade(l), Rc::downgrade(r));
    let op = Some(Operation::Mul(l_rc, r_rc));
    Unit::new(_Unit::new(
        result,
        op,
        vec![l.clone(), r.clone()]
    ))
}


pub fn pow(l: &Unit, r: f32) -> Unit {
    let result = l.borrow().data.powf(r);
    let l_rc = Rc::downgrade(l);
    let op = Some(Operation::Pow(l_rc, r));
    Unit::new(_Unit::new(
        result,
        op,
        vec![l.clone()]
    ))
}

pub fn sub(l: &Unit, r: &Unit) -> Unit {
    add(l, &mul(r, &new_unit(-1.0)))
}

pub fn div(l: &Unit, r: &Unit) -> Unit {
    assert!(r.borrow().data != 0.0);
    mul(l, &pow(r, -1.0))
}

pub fn tanh(x: &Unit) -> Unit {
    let result = x.borrow().data.tanh();
    let x_rc = Rc::downgrade(x);
    let op = Some(Operation::Tanh(x_rc));
    Unit::new(_Unit::new(
        result,
        op,
        vec![x.clone()]
    ))
    // let mut output = _Unit::new(x.borrow().data.tanh());
    // let x_rc = Rc::downgrade(x);
    // output.operation = Some(Operation::Tanh(x_rc));
    // output.children.push(x.clone());
    // output.to_unit()
}


pub fn relu(x: &Unit) -> Unit {
    let result = if x.borrow().data > 0.0 { x.borrow().data } else { 0.0 };
    let x_rc = Rc::downgrade(x);
    let op = Some(Operation::ReLU(x_rc));
    Unit::new(_Unit::new(
        result,
        op,
        vec![x.clone()]
    ))
    
    // let mut output = _Unit::new(if x.borrow().data > 0.0 { x.borrow().data } else { 0.0 });
    // let x_rc = Rc::downgrade(x);
    // output.operation = Some(Operation::ReLU(x_rc));
    // output.children.push(x.clone());
    // output.to_unit()
}

pub fn backward(u: &Unit) {
    u.borrow_mut().grad = 1.0;
    let topo_n = topological_sort_circle(u);


    if topo_n.is_none() {
        println!("Cyclic graph detected");
        return;
    }
    let mut topo_real = topo_n.unwrap();
    topo_real.reverse();
    println!("Topological sort done, found {}", topo_real.len());
    for bu in topo_real.iter(){
        //bu.upgrade().unwrap().borrow_mut().self_back_propagation();
        bu.borrow_mut().self_back_propagation();
    }
}


use std::collections::{HashMap, HashSet, VecDeque};
fn topological_sort(graph: &HashMap<usize, Vec<usize>>) -> Option<Vec<usize>> {
    let mut in_degrees = HashMap::new();
    let mut queue = VecDeque::new();
    let mut sorted = Vec::new();
    // 计算每个顶点的入度
    for (&node, neighbors) in graph {
        in_degrees.entry(node).or_insert(0);
        for &neighbor in neighbors {
            *in_degrees.entry(neighbor).or_insert(0) += 1;
        }
    }
    // 将所有入度为0的顶点加入队列
    for (&node, &in_degree) in in_degrees.iter() {
        if in_degree == 0 {
            queue.push_back(node);
        }
    }
    // Kahn算法主循环
    while let Some(node) = queue.pop_front() {
        sorted.push(node);
        if let Some(neighbors) = graph.get(&node) {
            for &neighbor in neighbors {
                let entry = in_degrees.get_mut(&neighbor)?;
                *entry -= 1;
                if *entry == 0 {
                    queue.push_back(neighbor);
                }
            }
        }
    }
    // 检查是否存在环
    if sorted.len() == in_degrees.len() {
        Some(sorted)
    } else {
        None
    }
}

pub fn topological_sort_circle(node: &Unit) -> Option<Vec<Unit>> {
    let mut visited = HashSet::new();
    let mut stack = HashSet::new(); // Track nodes currently on the stack
    let mut result = Vec::new();

    if detect_cycle(node, &mut visited, &mut stack, &mut result) {
        return None; // Graph contains cycle
    }

    Some(result)
}

fn detect_cycle(
    node: &Unit,
    visited: &mut HashSet<usize>,
    stack: &mut HashSet<usize>,
    result: &mut Vec<Unit>,
) -> bool {
    let id = node.borrow().id();

    if stack.contains(&id) {
        return true; // Cycle detected
    }

    if visited.contains(&id) {
        return false; // Node already visited and no cycle detected
    }

    stack.insert(id); // Mark node as being on the stack
    visited.insert(id);

    for child in &node.borrow().children {
        if detect_cycle(child, visited, stack, result) {
            return true; // Propagate cycle detection
        }
    }

    stack.remove(&id); // Remove node from stack after processing
    result.push(node.clone()); // Add node to result after processing
    false
}

fn rev_topological_sort_dfs(u:&Unit) -> Option<Vec<Unit>> {
    let mut visited: HashSet<usize> = HashSet::new();
    let mut sorted = Vec::new();
    let mut stack = VecDeque::new();
    let mut on_stack = HashSet::new();

    stack.push_back(u.clone());

    while let Some(current) = stack.pop_back() {
        let id = current.borrow().id();

        // if on_stack.contains(&id) {
        //     // cycle detected
        //     return None;
        // }

        if visited.contains(&id) {
            continue;
        }
        on_stack.insert(id);
        visited.insert(id);
        for child in current.borrow().children.iter() {
            stack.push_back(child.clone());
            // on_stack.insert(child.borrow().id());
        }
        sorted.push(current.clone());
    }

    //sorted.reverse();
    Some(sorted)
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::unit::Operation;


    #[test]
    fn test_back_propagation() {
        let a = new_unit(1.0);
        let b = new_unit(2.0);
        let mut c = add(&a, &b);
        c.borrow_mut().grad = 1.0;
        c.borrow_mut().self_back_propagation();
        println!("{:?}", c.borrow());
        assert_eq!(a.borrow().grad, 1.0);
        assert_eq!(b.borrow().grad, 1.0);
    }

    use std::collections::{HashMap, HashSet, VecDeque};

    #[test]
    fn test_topo() {
        let graph = vec![
            (2, vec![3, 0, 1]),
            (3, vec![1]),
            (0, vec![1]),
            (1, vec![4]),
        ].iter().cloned().collect::<HashMap<_, _>>();
        match topological_sort(&graph) {
            Some(sorted) => println!("Topologically sorted: {:?}", sorted),
            None => println!("No topological sort exists due to cycles in the graph."),
        }
    }

    #[test]
    fn test_topo_dfs() {
        let a = new_unit(1.0);
        let b = new_unit(2.0);
        let c = add(&a, &b); // 3
        let d = new_unit(4.0); // 2
        let e = tanh(&c); // 0.99505475
        let f = add(&d, &e); // 2.99505475
        //let mut sorted = rev_topological_sort_dfs(&f).unwrap();
        let mut sorted = topological_sort_circle(&f).unwrap();
        sorted.reverse();
        for  s in sorted.iter() {
            println!("{:}", s.borrow());
        }

        backward(&f);

        for ref s in sorted {
            println!("{:}", s.borrow());
        }
    }

    #[test]
    fn test_cyclic_compute_graph() {
        let a = new_unit(1.0);
        let b = new_unit(2.0);
        let c = add(&a, &b); // 3
        let d = new_unit(4.0); // 2
        let e = tanh(&c); // 0.99505475
        let f = add(&d, &e); // 2.99505475
        let g = add(&f, &f); // 5.9901095
        let mut sorted = rev_topological_sort_dfs(&g);
        assert!(sorted.is_none());
    }

    #[test]
    fn float_pow() {
        let a = new_unit(2.0);
        let c = pow(&a, -1.0);
        println!("{:?}", c.borrow());

    }

}
