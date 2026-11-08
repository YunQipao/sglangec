import copy,unittest
from .online_rs_identity import identity_for,validate_identity,stripe_id

class PrefixIdentityTests(unittest.TestCase):
    def rank(self,count):return dict(chunk_id=0,token_start=0,token_count=count,token_range=[0,count],layer_range=[0,36],layers=list(range(36)))
    def test_separate_bounded_prefix_scope(self):
        for n in (256,512,1024,2048,4096):
            with self.subTest(n=n):
                rank=self.rank(n);identity=identity_for('run','pair','version',rank,full_prefix=True)
                validate_identity(identity,rank)
                self.assertEqual(identity['format'],'kv-cold-rs-prefix-identity-v1')
                if n>256:
                    with self.assertRaises(ValueError):identity_for('run','pair','version',rank)
        rank=self.rank(256)
        self.assertNotEqual(stripe_id(identity_for('run','pair','v',rank)),stripe_id(identity_for('run','pair','v',rank,full_prefix=True)))
    def test_prefix_range_and_type_rejections(self):
        for n in (0,255,513,768,8192,True):
            with self.subTest(n=n),self.assertRaises(ValueError):identity_for('r','p','v',self.rank(n),full_prefix=True)
        r=self.rank(1024);i=identity_for('r','p','v',r,full_prefix=True)
        for field,value in [('chunk_id',1),('token_range',[0,256]),('layer_range',[0,35]),('format','unknown')]:
            x=copy.deepcopy(i);x[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):validate_identity(x,r)
        r['token_count']=4096
        with self.assertRaises(ValueError):validate_identity(i,r)
