use std::ops::{Add, Div};

use pyo3::types::{PyInt, PyIterator};

use {
    pyo3::{
        prelude::*,
        types::{PyFunction, PyList},
    },
    rand::seq::IteratorRandom,
};

trait Median {
    type Output;
    fn median(self: Self) -> Option<Self::Output>;
}

impl<Item> Median for Vec<Item>
where
    Item: Add<Item, Output = Item>, // Needed to add items
    Item: Copy,                     // Needed to move items.
    Item: Div<Item, Output = Item>, // Needed to divide items
    Item: From<u8>,                 // Needed to divide by Item::from(2)
    Item: Ord,                      // Needed for .sort()
{
    type Output = Item;

    fn median(self: Self) -> Option<Self::Output> {
        let mut v: Vec<Item> = self.iter().cloned().collect();
        v.sort();

        let l = v.len() as f64;
        if l % 2. == 0. {
            return Some(v[((l / 2.) - 1.) as usize]);
        } else if l % 2. != 0. {
            let lhs = v[((l / 2.).floor() - 1.) as usize];
            let rhs = v[((l / 2.).floor()) as usize];
            let med = (lhs + rhs).div(Item::from(2));
            return Some(med);
        }
        None
    }
}

#[pyclass(unsendable)]
pub struct Node {
    point: Py<PyAny>,
    threshold: i64,
    inside: Option<Box<Node>>,
    outside: Option<Box<Node>>,
}

#[pymethods]
impl Node {
    #[new]
    pub fn new(point: &PyAny, dist_fn: &PyFunction, subitems: &PyList) -> PyResult<Self> {
        let mut n = Node {
            point: point.to_object(subitems.py()),
            threshold: 0,
            inside: None,
            outside: None,
        };

        if subitems.len() == 0 {
            n.threshold = 0;
            return Ok(n);
        }

        let mut distances: Vec<(i64, &PyAny)> = Vec::new();
        for d in subitems {
            let dist = dist_fn.call1((&n.point, d))?;
            let dd = dist.extract::<i64>()?;
            distances.push((dd, d));
        }

        let mut just_distances: Vec<i64> = Vec::new();
        for d in distances.iter() {
            just_distances.push(d.0);
        }
        n.threshold = unsafe { just_distances.median().unwrap_unchecked() };

        // TODO: perhaps we can find a way to use PyList
        // for both of these since they have to be converted
        // into it eventually
        let mut inside: Vec<&PyAny> = Vec::new();
        let mut outside: Vec<&PyAny> = Vec::new();
        for d in distances {
            if d.0 <= n.threshold {
                inside.push(d.1);
                continue;
            } else if d.0 > n.threshold {
                outside.push(d.1);
                continue;
            }
        }

        if !inside.is_empty() {
            let i = unsafe {
                (0..inside.len())
                    .choose(&mut rand::thread_rng())
                    .unwrap_unchecked()
            };
            let choice = inside.swap_remove(i);
            let ctx = choice.py();
            n.inside = Some(Box::new(Node::new(
                choice,
                dist_fn,
                PyList::new(ctx, inside),
            )?));
        }

        if !outside.is_empty() {
            let i = unsafe {
                (0..outside.len())
                    .choose(&mut rand::thread_rng())
                    .unwrap_unchecked()
            };
            let choice = outside.swap_remove(i);
            let ctx = choice.py();
            n.outside = Some(Box::new(Node::new(
                choice,
                dist_fn,
                PyList::new(ctx, outside),
            )?));
        }

        Ok(n)
    }

    pub fn insert(&mut self, point: &PyAny, dist_fn: &PyFunction) -> PyResult<()> {
        let dist_calc = dist_fn.call1((&self.point, point))?;
        let distance = dist_calc.extract::<i64>()?;

        if distance < self.threshold {
            match &mut self.inside {
                Some(ref mut bn) => {
                    bn.insert(point, dist_fn)?;
                }
                None => {
                    self.inside = Some(Box::new(Node::new(
                        point,
                        dist_fn,
                        PyList::empty(dist_fn.py()),
                    )?));
                }
            };
            return Ok(());
        }

        match &mut self.outside {
            Some(ref mut bn) => {
                bn.insert(point, dist_fn)?;
            }
            None => {
                self.outside = Some(Box::new(Node::new(
                    point,
                    dist_fn,
                    PyList::empty(dist_fn.py()),
                )?));
            }
        }

        Ok(())
    }

    /*pub fn __len__(&self) -> PyResult<i64> {
        let insides = match self.inside {
            Some(bn) => bn.__len__()? + 1,
            None => 0,
        };
        let outsides = match self.outside {
            Some(bn) => bn.__len__()? + 1,
            None => 0,
        };
        Ok(insides + outsides)
    }*/

    // DANGER: disgusting memory leak, but how else do I format a string
    // and get a static reference to it?
    pub fn __repr__(&self) -> &'static str {
        return Box::leak(
            format!("Node(point={}, threshold={})", self.point, self.threshold).into_boxed_str(),
        );
    }
}

#[pyclass]
pub struct VPTree {
    vantage_point: Option<Node>,
    threshold: i64,
    dist_fn: Py<PyFunction>,
}

#[pymethods]
impl VPTree {
    #[new]
    pub fn new(dist_fn: &PyFunction, points: Option<&PyList>) -> PyResult<Self> {
        let ctx = dist_fn.py();

        let mut vptree = VPTree {
            vantage_point: None,
            threshold: 0,
            dist_fn: dist_fn.into_py(ctx),
        };

        // TODO: it would be faster to not have to convert this to
        // a native rust type, then have to convert it back to
        // a python type, but I really need swap_remove
        let mut points = match points {
            Some(p) => p.extract::<Vec<&PyAny>>()?,
            None => return Ok(vptree),
        };

        if points.len() == 0 {
            return Ok(vptree);
        }

        let i = unsafe {
            (0..points.len())
                .choose(&mut rand::thread_rng())
                .unwrap_unchecked()
        };
        let choice = points.swap_remove(i);

        vptree.vantage_point = Some(Node::new(choice, dist_fn, PyList::new(ctx, points))?);

        Ok(vptree)
    }

    pub fn knn(&self, query: &PyAny, k: i64) -> PyResult<Py<PyList>> {
        let ctx = query.py();

        match self.vantage_point {
            Some(_) => {}
            None => return Ok(PyList::empty(ctx).into_py(ctx)),
        }

        let mut tau = f64::INFINITY;
        let mut to_search = Vec::with_capacity(1);
        to_search.push(unsafe { self.vantage_point.as_ref().unwrap_unchecked() });

        let mut results: Vec<(&PyAny, i64)> = vec![];

        while !to_search.is_empty() {
            let current_node = unsafe { to_search.pop().unwrap_unchecked() };
            let dist_calc = self
                .dist_fn
                .call1(ctx, (query, current_node.point.as_ref(ctx)))?;
            let distance = dist_calc.extract::<i64>(ctx)?;

            if (distance as f64) < tau {
                results.push((current_node.point.as_ref(ctx), distance));
            }

            if results.len() > (k as usize) {
                results.sort_by_key(|key| key.1);
                results.pop();
                let tau_calc = self.dist_fn.call1(ctx, (query, results[results.len()-1].0))?;
                tau = (tau_calc.extract::<i64>(ctx)?) as f64;
            }

            if (distance as f64) < (current_node.threshold as f64) + tau {
                match &current_node.outside {
                    Some(bn) => to_search.push(&bn),
                    None => {}
                }
            }

            if (distance as f64) >= (current_node.threshold as f64) - tau {
                match &current_node.inside {
                    Some(bn) => to_search.push(&bn),
                    None => {}
                }
            }
        }

        return Ok(PyList::new(ctx, results).into_py(ctx));
    }
}

#[pymodule]
pub fn vptrees(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Node>()?;
    m.add_class::<VPTree>()?;

    Ok(())
}
