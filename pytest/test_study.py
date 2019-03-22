
import pytest

# @pytest.fixture
# def username():
#     print("hi!!!")
#     return 'username'
#
# @pytest.fixture
# def other_username(username):
#     print("hi2!!!")
#     return 'other-' + username
#
#
# @pytest.mark.parametrize('username', ['directly-overridden-username'])
# def test_username(username):
#     assert username == 'directly-overridden-username'
#
# @pytest.mark.parametrize('username', ['directly-overridden-username-other'])
# def test_username_other(other_username):
#     assert other_username == 'other-directly-overridden-username-other'


# @pytest.fixture(params=[0, 200], ids=["spam", "ham"])
# def a(request):
#     return request.param
#
# def test_a(a):
#     # pass
#     print(a)
#
# def idfn(fixture_value):
#     if fixture_value == 0:
#         return "eggs"
#     else:
#         return None
#
# @pytest.fixture(params=[0, 40], ids=idfn)
# def b(request):
#     return request.param
#
# def test_b(b):
#     # pass
#     print(b)


@pytest.fixture(scope="class",params=[[1,2,3],[4,5,6],[7,8,9]])
def ali(request):
    yield request.param

# @pytest.fixture(scope="class",autouse=True)
# def bed():
#     print("hi")

# @pytest.fixture()
# def a(all): return all[0]
#
# @pytest.fixture()
# def b(all): return all[1]
#
# @pytest.fixture()
# def c(all): return all[2]


# @pytest.mark.usefixtures("all")
class Test_A:

    @pytest.fixture()
    def a(self,ali): return ali[0]

    @pytest.fixture()
    def b(self,ali): return ali[1]

    @pytest.fixture()
    def c(self,ali): return ali[2]


    def test_one(self,a,b):
        print(a+b)
        print("test_1")
        pass


    def test_two(self,a,c):
        print(a+c)
        print("test_2")
        pass

# def test_temp(a):
#     print(a)
